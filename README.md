# SGLang-FluentLLM

The LongCat series models have consistently followed the principle of Model–System Co-Design, which introduces unique challenges for both the training and inference systems. To help the community better adopt and use LongCat models, we are open-sourcing part of our inference engine (SGLang-FluentLLM) as well as several key kernels.

## Engine
Our inference engine is built on top of the [SGLang](https://github.com/sgl-project/sglang) codebase, with the following enhanced capabilities:

- Refactored the speculative decoding workflow to make it compatible with overlap scheduling
- Combined Target + Verify + Draft into a single CUDA graph to reduce speculative decoding overhead
- Support for Eagle, MTP, and PLD style speculative decoding
- Layer-wise KVCache transfer, overlapping prefill computation with KVCache communication
- Decode Radix Tree Cache to reduce KVCache transfer volume between PDs


We sincerely appreciate the solid work and inspiration brought by the SGLang community.

## Kernels
On the kernels side, we are open-sourcing:

- [FlashMLA SwapAB](https://github.com/meituan-longcat/FlashMLA/tree/feature/swapAB) optimizations
- [FlashMLA FP8 KVCache + FP8 Compute](https://github.com/meituan-longcat/FlashMLA/tree/feature/ckv_fp8_per_token) optimizations
  - This optimization is detailed in the paper [**SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining**](https://arxiv.org/pdf/2602.10718).
- [DeepGemm SwapAB Offset + PDL](https://github.com/meituan-longcat/DeepGEMM/tree/feature/swap_ab) optimizations
- Communication–computation fused kernels optimizations in [FlashInfer](https://github.com/meituan-longcat/flashinfer/tree/feature/longcat_main)

We would also like to thank the broader LLM inference community. It is an honor for us to grow together with this community.

## Note
- We use [Dynamo](https://github.com/ai-dynamo/dynamo) for KVCache-aware request scheduling. As a result, in SGLang-FluentLLM we have removed SGLang’s sgl-model-gateway.
- For multimodal models, we adopt a decoupled architecture that differs from the one used in the SGLang community. Therefore, multimodal support has also been removed from SGLang-FluentLLM itself (even in our internal setup, SGLang-FluentLLM is still used as the LLM backbone for multimodal inference).
- Tested on Nvidia GPUs H800/H20.

## How to Use

Please refer to [Quick Start](https://github.com/meituan-longcat/SGLang-FluentLLM/blob/main/Quick_Start.md)
