# SGLang-FluentLLM

The LongCat series models have consistently followed the principle of Model–System Co-Design, which introduces unique challenges for both the training and inference systems. To help the community better adopt and use LongCat models, we are open-sourcing part of our inference engine (SGLang-FluentLLM) as well as several key kernels.

## Engine
Our inference engine is built on top of the [SGLang](https://github.com/sgl-project/sglang) codebase, with the following enhanced capabilities:

- Refactored the speculative decoding workflow to make it compatible with overlap scheduling
- Combined Target + Verify + Draft into a single CUDA graph to reduce speculative decoding overhead
- Support for Eagle, MTP, and PLD style speculative decoding
- Layer-wise KVCache transfer, overlapping prefill computation with KVCache communication
- Decode Radix Tree Cache to reduce KVCache transfer volume between PDs
- LongCat 2.5 W4A4 multimodal inference on Ascend A2. See the [NPU deployment guide](npu_test/ReadMe.md).


We sincerely appreciate the solid work and inspiration brought by the SGLang community.

## Kernels
On the kernels side, we are open-sourcing:

- FlashMLA SwapAB optimizations
- FlashMLA FP8 KVCache + FP8 Compute optimizations
- DeepGemm SwapAB Offset + PDL optimizations
- Communication–computation fused kernels optimizations in FlashInfer

We would also like to thank the broader LLM inference community. It is an honor for us to grow together with this community.

## Note
- We use [Dynamo](https://github.com/ai-dynamo/dynamo) for KVCache-aware request scheduling. As a result, in SGLang-FluentLLM we have removed SGLang’s sgl-model-gateway.
- LongCat 2.5 W4A4 multimodal inference uses an Encoder–Prefill–Decode (EPD) architecture: a standalone Encoder service handles visual encoding, while SGLang-FluentLLM runs the language model on Prefill/Decode nodes.
- Tested on Nvidia GPUs H800/H20.
- Tested on the Ascend A2.
