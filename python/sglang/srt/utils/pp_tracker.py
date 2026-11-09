import json
import logging
import threading
import time
from datetime import datetime

import zmq

from sglang.srt.managers.scheduling.schedule_batch import PrefillReqInfo

logger = logging.getLogger(__name__)


class IdGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.id = 0

    def generate_id(self):
        with self.lock:
            self.id += 1
            return self.id


class PPBatchMetric:
    def __init__(self, it, timings_dict, pp0_recv_time=0.0):
        self.it = it
        self.timings_dict = timings_dict
        self.pp0_recv_time = pp0_recv_time


class PPBatchMetricAggregator:
    def __init__(self, batch_info):
        self.batch_info: list[PrefillReqInfo] = batch_info
        self.pp_batch_metrics: dict[int, PPBatchMetric] = {}

    def __repr__(self):
        parts = []
        for rank in sorted(self.pp_batch_metrics.keys()):
            metric = self.pp_batch_metrics[rank]
            detail = " ".join(f"{k}={v:.1f}ms" for k, v in metric.timings_dict.items())
            detail += f" pp0_recv_time={datetime.fromtimestamp(metric.pp0_recv_time).strftime('%H:%M:%S.%f')[:-3]}"
            parts.append(f"PP{rank}: [{detail}]")

        sum_q = sum(info.extend_input_len for info in self.batch_info) if self.batch_info else 0
        sum_kv = sum(info.context_len for info in self.batch_info) if self.batch_info else 0
        sum_qkv = sum(info.extend_input_len * info.context_len for info in self.batch_info) if self.batch_info else 0
        sum_host_hit = sum(info.host_hit_len for info in self.batch_info) if self.batch_info else 0
        sum_storage_hit = sum(info.storage_hit_len for info in self.batch_info) if self.batch_info else 0

        sum_kv = float(sum_kv) / 1024
        sum_qkv = float(sum_qkv) / 1024 / 1024
        predict_time = float(sum_q) * 0.0125 + float(sum_kv) * 0.118 + float(sum_qkv) * 0.078
        stats = f"q={sum_q} kv={sum_kv:.2f}K qkv={sum_qkv:.2f}M l2_hit={sum_host_hit} l3_hit={sum_storage_hit}, predict_time={predict_time:.2f}"

        return f"{self.batch_info}, {stats}, " + " | ".join(parts)


class PPTracker:
    def __init__(self, pp_rank, pp_size, pp0_ip, pp0_port):
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp0_ip = pp0_ip
        self.pp0_port = pp0_port

        self._shutdown = threading.Event()

        self.lock = threading.Lock()
        if pp_rank == 0:
            # hicache queue tracker: l2_write、l2_load、l3_load、l3_write、l3_release
            self.hicache_queue_tracker: list[list[int]] = [[0] * pp_size for _ in range(5)]

            # request_id -> transferred pp_ranks
            self.transfer_request_tracker: dict[int, set[int]] = {}

            # batch info tracker: iteration -> {pp_rank -> batch_metric}
            self.batch_metric_tracker: dict[int, dict[int, PPBatchMetric]] = {}
            self.batch_metric_aggregator: dict[int, PPBatchMetricAggregator] = {}

            self.context = zmq.Context()
            self.server_socket = self.context.socket(zmq.PULL)
            self.server_socket.bind(f"tcp://{pp0_ip}:{pp0_port}")

            self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self._server_thread.start()

            logger.info(f"PPTracker server started on {pp0_ip}:{self.pp0_port}")
        else:
            self.context = zmq.Context()
            self.client_socket = self.context.socket(zmq.PUSH)
            self.client_socket.connect(f"tcp://{pp0_ip}:{pp0_port}")

            logger.info(f"PPTracker client connected to {pp0_ip}:{pp0_port}")

    def _server_loop(self):
        """PP rank 0 的后台接收线程，持续接收其他 rank 的进度更新"""
        while not self._shutdown.is_set():
            try:
                if self.server_socket.poll(timeout=100):
                    msg = self.server_socket.recv_string()
                    data = json.loads(msg)
                    pp_rank = data["pp_rank"]
                    type = data["type"]

                    with self.lock:
                        if type == "batch_metric":
                            iteration = data["iteration"]
                            timings = data["timings"]
                            if iteration not in self.batch_metric_tracker:
                                self.batch_metric_tracker[iteration] = {}
                            self.batch_metric_tracker[iteration][pp_rank] = PPBatchMetric(iteration, timings, time.time())
                            # Check if all PP ranks have reported for this iteration
                            if len(self.batch_metric_tracker[iteration]) == self.pp_size:
                                self.batch_metric_aggregator[iteration].pp_batch_metrics = self.batch_metric_tracker[iteration]
                                del self.batch_metric_tracker[iteration]

                    logger.debug(f"PPTracker Received {data}")
            except zmq.ZMQError:
                if self._shutdown.is_set():
                    break
                raise

    def check_batch_metrics(self):
        """Return completed batch metrics (pp_batch_metrics not empty) and remove them from tracker."""
        batch_metrics = {}
        with self.lock:
            done_its = [it for it, agg in self.batch_metric_aggregator.items() if agg.pp_batch_metrics]
            for it in done_its:
                batch_metrics[it] = self.batch_metric_aggregator.pop(it)
            return batch_metrics

    def update_batch_metric(self, iteration: int, timings: dict, batch_info=None):
        """Send batch timing info from this PP rank to PP0."""
        if self.pp_rank > 0:
            msg = json.dumps(
                {
                    "pp_rank": self.pp_rank,
                    "type": "batch_metric",
                    "iteration": iteration,
                    "timings": timings,
                }
            )
            try:
                self.client_socket.send_string(msg)
            except zmq.ZMQError:
                logger.info(f"PPTracker send batch_metric failed: {msg}")
        else:
            with self.lock:
                if iteration not in self.batch_metric_tracker:
                    self.batch_metric_tracker[iteration] = {}
                if iteration not in self.batch_metric_aggregator:
                    self.batch_metric_aggregator[iteration] = PPBatchMetricAggregator(batch_info)

                self.batch_metric_tracker[iteration][0] = PPBatchMetric(iteration, timings, time.time())
                # Check if all PP ranks have reported for this iteration
                if len(self.batch_metric_tracker[iteration]) == self.pp_size:
                    self.batch_metric_aggregator[iteration].pp_batch_metrics = self.batch_metric_tracker[iteration]
                    del self.batch_metric_tracker[iteration]

    def shutdown(self):
        """关闭 ZMQ 连接和后台线程"""
        self._shutdown.set()
        if self.pp_rank == 0:
            if hasattr(self, "_server_thread"):
                self._server_thread.join(timeout=5)
            self.server_socket.close(linger=0)
        else:
            self.client_socket.close(linger=0)
        self.context.term()
        logger.info("PPTracker shut down")
