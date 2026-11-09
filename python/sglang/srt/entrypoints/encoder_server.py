"""Encoder Server CLI entrypoint for EPD (Encode-Prefill-Decode) disaggregation.

This server supports two launch modes:

1. **Multi-process TP mode** (``--attn-tp-size N`` with ``--nprocs-per-node N``):
   The main process spawns N worker processes (one per TP rank) via
   ``multiprocessing.Process``. Each worker constructs its own ``ModelRunner``
   (which internally calls ``init_distributed_environment`` +
   ``initialize_model_parallel``), and runs an ``EncoderServer``. Only rank 0
   starts the HTTP/gRPC server; all ranks run the EncodeScheduler step loop so
   that ViT forward's ``RowParallelLinear`` all_reduce is satisfied.

2. **Single-process mode** (``--nprocs-per-node 1``, default):
   Backward-compatible with the original lightweight server — one process loads
   the model and serves HTTP directly. Used for TP=1 deployments.

Unlike the standard SGLang server (which runs TokenizerManager/Scheduler/Detokenizer
in 3 processes), the Encoder server runs everything in a single process per TP rank
because:
- It only does ViT forward (no LM, no KV cache, no sampling).
- The EncodeScheduler is lightweight (just a sorted queue).
- No detokenization needed (output is rvq_indices or vit_embedding tensors).

Usage (TP=2 example):
    python -m sglang.srt.entrypoints.encoder_server \\
        --model-path <model> \\
        --disaggregation-mode encode \\
        --nnodes 1 --nprocs-per-node 2 --attn-tp-size 2 \\
        --dist-init-addr 127.0.0.1:3000 \\
        --encoder-output-format rvq_indices \\
        --encoder-http-port 9101 \\
        --tp 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

# Fix a bug of Python threading
import threading
threading._register_atexit = lambda *args, **kwargs: None

import torch
import uvloop

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def parse_args() -> ServerArgs:
    """Parse command-line arguments into ServerArgs with encoder_only=True enforced."""
    parser = argparse.ArgumentParser(
        description="EPD Encoder Server: runs ViT only and serves encode_async via HTTP (and optional gRPC)."
    )
    # Use ServerArgs' own argparse integration
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Build ServerArgs from parsed CLI args
    server_args = ServerArgs.from_cli_args(args)

    # Enforce encoder-only mode
    server_args.encoder_only = True
    server_args.enable_multimodal = True

    # The Encoder doesn't need HTTP serving, radix cache, or sampling
    server_args.disable_radix_cache = True

    return server_args


def load_encoder_model(
    server_args: ServerArgs,
    gpu_id: int = 0,
    attn_tp_rank: int = 0,
    global_rank: int = 0,
    nccl_port: int = 0,
):
    """Load the model in encoder_only mode.

    Returns the ``ModelRunner`` (with ``.model`` being the model object, e.g.
    ``LongcatFlashVLForConditionalGeneration``) with only vision_tower + bridge +
    quantizer initialized.

    Args:
        server_args: ServerArgs with encoder_only=True.
        gpu_id: GPU device ID for this TP rank.
        attn_tp_rank: attention TP rank of this worker (0..attn_tp_size-1).
        global_rank: global distributed rank of this worker.
        nccl_port: NCCL distributed init port (0 → auto from dist_init_addr).

    Returns:
        model_runner: The ModelRunner with encoder components initialized.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

    logger.info(
        "Loading model in encoder_only mode: %s (attn_tp_rank=%d, global_rank=%d)",
        server_args.model_path, attn_tp_rank, global_rank,
    )

    model_config = ModelConfig(
        server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        context_length=server_args.context_length,
        model_override_args=server_args.json_model_override_args,
        is_embedding=False,
        dtype=server_args.dtype,
        quantization=server_args.quantization,
        server_args=server_args,
    )

    # Create a minimal ModelRunner just for model loading
    # (We don't need the full ModelRunner for KV cache / forward batch)
    # ModelRunner.__init__ → init_torch_distributed → init_distributed_environment
    # + initialize_model_parallel (NCCL init is unconditional, even in encoder_only).
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=server_args.attn_tp_size,
        world_size=server_args.world_size,
        moe_ep_rank=0,
        moe_ep_size=server_args.ep_size,
        global_rank=global_rank,
        nccl_port=nccl_port,
        server_args=server_args,
    )

    logger.info(
        "Encoder model loaded: %s (encoder_only=True, attn_tp_rank=%d)",
        model_config.model_path, attn_tp_rank,
    )
    return model_runner


async def run_encoder_server(server_args: ServerArgs) -> None:
    """Main async entry (single-process mode): load model → init EncoderServer → serve.

    This path is used when ``attn_tp_size == 1``. For ``attn_tp_size > 1`` the
    multi-process spawn path in ``launch_encoder_server`` is used instead.
    """
    from sglang.srt.disaggregation.encode.server import EncoderServer

    # 1. Load model in encoder_only mode (returns ModelRunner)
    model_runner = load_encoder_model(server_args)

    # 2. Create EncoderServer
    encoder_server = EncoderServer(server_args=server_args, model_runner=model_runner)

    # 3. Start serving (HTTP always, gRPC optional when stubs are available)
    http_port = server_args.encoder_http_port
    grpc_port = server_args.encoder_grpc_port  # None → skip gRPC
    logger.info(
        "Starting Encoder server: http_port=%d, grpc_port=%s (output_format=%s)",
        http_port,
        grpc_port,
        server_args.encoder_output_format,
    )
    await encoder_server.serve(http_port=http_port, grpc_port=grpc_port)

    logger.info("Encoder server is ready.")

    # 4. Wait until shutdown
    try:
        await asyncio.Event().wait()  # block forever
    except asyncio.CancelledError:
        pass
    finally:
        await encoder_server.shutdown()
        logger.info("Encoder server shut down.")


def run_encoder_worker_process(
    server_args: ServerArgs,
    gpu_id: int,
    attn_tp_rank: int,
    global_rank: int,
    nccl_port: int,
    pipe_writer,
) -> None:
    """Each Encoder TP worker process entry point.

    1. Constructs ``ModelRunner`` (internally calls ``init_distributed_environment``
       + ``initialize_model_parallel`` for NCCL init).
    2. Constructs ``EncoderServer`` (HTTP + step loop). Only ``attn_tp_rank==0``
       starts the HTTP/gRPC server; all ranks run the step loop because ViT
       forward's ``RowParallelLinear`` triggers NCCL ``all_reduce`` that must be
       participated in by every rank.
    3. Signals ``{"status": "ready"}`` back to the spawner via ``pipe_writer``,
       then blocks forever (until SIGTERM/SIGINT).
    """
    import setproctitle
    from sglang.srt.disaggregation.encode.server import EncoderServer

    setproctitle.setproctitle(f"sglang::encoder_worker rank{attn_tp_rank}")
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logger.info(
        "[encoder_worker rank=%d] starting: gpu_id=%d, global_rank=%d, nccl_port=%d",
        attn_tp_rank, gpu_id, global_rank, nccl_port,
    )

    # 1. Load model in encoder_only mode (returns ModelRunner with NCCL init done)
    model_runner = load_encoder_model(
        server_args,
        gpu_id=gpu_id,
        attn_tp_rank=attn_tp_rank,
        global_rank=global_rank,
        nccl_port=nccl_port,
    )

    # 2. Create EncoderServer (all ranks construct it so step loop runs everywhere)
    encoder_server = EncoderServer(server_args=server_args, model_runner=model_runner)

    # 3. Start serving — HTTP only on rank 0 (control-plane single entry)
    http_port = server_args.encoder_http_port if attn_tp_rank == 0 else None
    grpc_port = server_args.encoder_grpc_port if attn_tp_rank == 0 else None

    async def _serve_and_signal():
        if http_port is not None:
            logger.info(
                "[encoder_worker rank=%d] starting HTTP server on port %d",
                attn_tp_rank, http_port,
            )
        await encoder_server.serve(http_port=http_port, grpc_port=grpc_port)
        pipe_writer.send({"status": "ready"})
        logger.info("[encoder_worker rank=%d] ready, blocking forever", attn_tp_rank)
        await asyncio.Event().wait()  # block forever

    try:
        loop.run_until_complete(_serve_and_signal())
    except asyncio.CancelledError:
        pass
    finally:
        loop.run_until_complete(encoder_server.shutdown())
        logger.info("[encoder_worker rank=%d] shut down", attn_tp_rank)
        loop.close()


def _wait_for_encoder_workers(procs, pipe_readers) -> None:
    """Exit the launcher when any worker exits, including during startup."""
    from multiprocessing.connection import wait

    pending = {reader: rank for rank, reader in enumerate(pipe_readers)}
    sentinels = {proc.sentinel: proc for proc in procs}
    exit_code = 1
    try:
        while pending:
            ready = wait([*pending, *sentinels])
            if any(handle in sentinels for handle in ready):
                raise RuntimeError("Encoder worker exited before all ranks were ready")
            for reader in ready:
                rank = pending.pop(reader)
                data = reader.recv()
                if data.get("status") != "ready":
                    raise RuntimeError(f"Unexpected ready signal from rank {rank}: {data}")
                logger.info("Encoder worker rank=%d reported ready", rank)

        logger.info("All %d encoder workers ready. Waiting for worker exit.", len(procs))
        exited = wait(list(sentinels))
        proc = sentinels[exited[0]]
        proc.join()
        exit_code = proc.exitcode if proc.exitcode and proc.exitcode > 0 else 1
        logger.error("Encoder worker %s exited (code=%s); stopping the TP group",
                     proc.name, proc.exitcode)
    except (EOFError, RuntimeError) as error:
        logger.error("Encoder worker startup failed: %s", error)
    finally:
        for reader in pipe_readers:
            reader.close()
        # Only stop children belonging to this Encoder instance. Bounded waits
        # prevent a stuck device thread from keeping the launcher alive.
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        deadline = time.monotonic() + 5.0
        for proc in procs:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
        for proc in procs:
            if proc.is_alive():
                proc.kill()
        deadline = time.monotonic() + 5.0
        for proc in procs:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
    raise SystemExit(exit_code)


def launch_encoder_server(server_args: ServerArgs) -> None:
    """Main process: spawn N encoder worker processes (N = attn_tp_size).

    Each worker runs ``run_encoder_worker_process``. The main process waits
    for all workers to signal "ready", then exits the complete TP group as
    soon as any worker exits.

    For ``attn_tp_size == 1`` this still spawns a single worker process for
    uniformity; the original single-process path (``run_encoder_server``) is
    kept for backward compatibility but not used when this function is invoked.
    """
    import multiprocessing as mp

    tp_size = server_args.attn_tp_size
    assert tp_size >= 1, f"attn_tp_size must be >= 1, got {tp_size}"

    # Determine a free NCCL init port. When --dist-init-addr is provided by the
    # user, ModelRunner will use it directly (nccl_port is ignored). Otherwise we
    # allocate a free TCP port on localhost for the gloo/nccl rendezvous.
    nccl_port = 0
    if not server_args.dist_init_addr:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            nccl_port = s.getsockname()[1]
        # Inject the port into server_args so ModelRunner's init_torch_distributed
        # picks it up (dist_init_method = f"tcp://127.0.0.1:{nccl_port}").
        server_args.dist_init_addr = f"127.0.0.1:{nccl_port}"
        logger.info("Auto-assigned dist_init_addr=%s for encoder TP", server_args.dist_init_addr)

    logger.info(
        "Launching %d encoder worker processes (attn_tp_size=%d, dist_init_addr=%s)",
        tp_size, tp_size, server_args.dist_init_addr,
    )

    mp.set_start_method("spawn", force=True)

    procs: list[mp.Process] = []
    pipe_readers: list = []
    for tp_rank in range(tp_size):
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = server_args.base_gpu_id + tp_rank * server_args.gpu_id_step
        global_rank = tp_rank  # single-node: global_rank == tp_rank
        proc = mp.Process(
            target=run_encoder_worker_process,
            args=(server_args, gpu_id, tp_rank, global_rank, nccl_port, writer),
            name=f"encoder_worker_rank{tp_rank}",
        )
        proc.start()
        writer.close()
        procs.append(proc)
        pipe_readers.append(reader)
        logger.info("Spawned encoder worker rank=%d pid=%d gpu_id=%d", tp_rank, proc.pid, gpu_id)

    _wait_for_encoder_workers(procs, pipe_readers)


def main() -> None:
    """CLI entrypoint.

    When ``attn_tp_size > 1``, spawns N worker processes via
    ``launch_encoder_server``. When ``attn_tp_size == 1``, runs the
    single-process path ``run_encoder_server`` directly (backward compatible).
    """
    server_args = parse_args()

    # Validate
    server_args.check_server_args()
    assert server_args.encoder_only, "Encoder server must run with encoder_only=True"
    logger.info("Encoder server args: %s", server_args)

    # Set up signal handler for graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Multi-process TP path: spawn N workers (each runs its own event loop)
    if server_args.attn_tp_size > 1:
        launch_encoder_server(server_args)
        return

    # Single-process path (TP=1): original lightweight server
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_encoder_server(server_args))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down...")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
