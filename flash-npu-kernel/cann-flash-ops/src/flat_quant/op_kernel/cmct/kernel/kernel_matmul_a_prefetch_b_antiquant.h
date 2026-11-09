/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_matmul_a_prefetch_b_antiquant.h
 * \brief
 */
#ifndef KERNEL_MATMUL_A_PREFETCH_B_ANTIQUANT_H
#define KERNEL_MATMUL_A_PREFETCH_B_ANTIQUANT_H

#include "../utils/integral_constant.h"
#include "../utils/tuple_utils.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "../prologue/constant.h"
#include "../utils/constant.h"
#include "../utils/math_utils.h"
#include "../utils/underscore.h"
#include "../utils/tensor_utils.h"
namespace Cmct::Gemm::Kernel {

using Cmct::Gemm::Get;
using AscendC::MakeCoord;
using AscendC::MakeLayout;
using AscendC::MakeShape;
using AscendC::MakeStride;

template <class ProblemShape_, class BlockMmad_, class BlockPrologue_, class BlockScheduler_>
class KernelMatmulAPrefetchBAntiquant {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockPrologue = BlockPrologue_;
    using BlockScheduler = BlockScheduler_;
    using BlockMmadArguments = typename BlockMmad::Arguments;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockPrologueArguments = typename BlockPrologue::Arguments;
    using BlockPrologueParams = typename BlockPrologue::Params;
    using BlockSchedulerArguments = typename BlockScheduler::Arguments;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmad;
        BlockPrologueArguments prologue;
        BlockSchedulerArguments scheduler;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmad;
        BlockPrologueParams prologue;
        BlockSchedulerParams scheduler;
    };

    template <typename Tiling>
    __aicore__ inline static Params ToUnderlyingArguments(const Arguments& args, Tiling const* tiling)
    {
        return {
            .problemShape = args.problemShape,
            .mmad = BlockMmad::ToUnderlyingArguments(args.problemShape, args.mmad, tiling),
            .prologue = BlockPrologue::ToUnderlyingArguments(args.problemShape, args.prologue, tiling),
            .scheduler = BlockScheduler::ToUnderlyingArguments(args.problemShape, args.scheduler, tiling)};
    }

    __aicore__ inline KernelMatmulAPrefetchBAntiquant() = default;
    __aicore__ inline KernelMatmulAPrefetchBAntiquant([[maybe_unused]] const Params& params){};

    template <int32_t CoreType = g_coreType>
    __aicore__ inline void operator()(const Params& params);

    template <>
    __aicore__ inline void operator()<AscendC::AIV>(const Params& params)
    {
        BlockPrologue blockPrologue(params.prologue);
        BlockScheduler blockScheduler(params.problemShape, params.scheduler);
        auto tensorB =
            MakeTensor((__gm__ typename BlockPrologue::ElementIn*)params.prologue.ptrB, params.prologue.layoutB);
        auto tensorScale = MakeTensor(
            (__gm__ typename BlockPrologue::ElementScale*)params.prologue.ptrScale, params.prologue.layoutScale);
        auto tensorOffset = MakeTensor(
            (__gm__ typename BlockPrologue::ElementScale*)params.prologue.ptrOffset, params.prologue.layoutScale);
        for (uint64_t coordM = blockScheduler.mStart; coordM < blockScheduler.mStop; coordM += blockScheduler.mStep) {
            auto tileM = Min(blockScheduler.mStop - coordM, blockScheduler.mTile);
            auto tileN = blockScheduler.n0Tile;
            for (uint64_t coordN = blockScheduler.n0Start; coordN < blockScheduler.n0Stop;
                 coordN += blockScheduler.n0Step) {
                auto blockScale = GetAntiScaleTile(tensorScale, coordN, tileN, params.prologue.antiQuantGroupSize);
                auto blockOffset = GetAntiScaleTile(tensorOffset, coordN, tileN, params.prologue.antiQuantGroupSize);
                auto blockWeight = GetWeightTile<WEIGHT_NZ>(tensorB, coordN, tileN, params.prologue.antiQuantGroupSize);
                blockPrologue(
                    blockWeight, blockScale, blockOffset, AscendC::MakeShape(tileN, Get<2>(params.problemShape)));
            }
            for (uint64_t coordN = blockScheduler.n1Start; coordN < blockScheduler.n1Stop;
                 coordN += blockScheduler.n1Step) {
                tileN = coordN + blockScheduler.n1Tile > blockScheduler.n1Stop ? blockScheduler.n1Stop - coordN :
                                                                                 blockScheduler.n1Tile;
                auto blockScale = GetAntiScaleTile(tensorScale, coordN, tileN, params.prologue.antiQuantGroupSize);
                auto blockOffset = GetAntiScaleTile(tensorOffset, coordN, tileN, params.prologue.antiQuantGroupSize);
                auto blockWeight = GetWeightTile<WEIGHT_NZ>(tensorB, coordN, tileN, params.prologue.antiQuantGroupSize);
                blockPrologue(
                    blockWeight, blockScale, blockOffset, AscendC::MakeShape(tileN, Get<2>(params.problemShape)));
            }
            for (uint64_t coordN = blockScheduler.n2Start; coordN < blockScheduler.n2Stop;
                 coordN += blockScheduler.n2Step) {
                tileN = coordN + blockScheduler.n2Tile > blockScheduler.n2Stop ? blockScheduler.n2Stop - coordN :
                                                                                 blockScheduler.n2Tile;
                auto blockScale = GetAntiScaleTile(tensorScale, coordN, tileN, params.prologue.antiQuantGroupSize);
                auto blockOffset = GetAntiScaleTile(tensorOffset, coordN, tileN, params.prologue.antiQuantGroupSize);
                auto blockWeight = GetWeightTile<WEIGHT_NZ>(tensorB, coordN, tileN, params.prologue.antiQuantGroupSize);
                blockPrologue(
                    blockWeight, blockScale, blockOffset, AscendC::MakeShape(tileN, Get<2>(params.problemShape)));
            }
        }
    }

    template <>
    __aicore__ inline void operator()<AscendC::AIC>(const Params& params)
    {
        auto tensorA = MakeTensor((__gm__ typename BlockMmad::ElementA*)(params.mmad.ptrA), params.mmad.layoutA);
        BlockMmad::PreloadA(tensorA, params.mmad);
        BlockMmad blockMmad(params.mmad);
        auto tensorC = MakeTensor((__gm__ typename BlockMmad::ElementC*)(params.mmad.ptrC), params.mmad.layoutC);
        auto tensorBias =
            MakeTensor((__gm__ typename BlockMmad::ElementBias*)(params.mmad.ptrBias), params.mmad.layoutBias);

        BlockScheduler blockScheduler(params.problemShape, params.scheduler);
        for (uint64_t coordM = blockScheduler.mStart; coordM < blockScheduler.mStop; coordM += blockScheduler.mStep) {
            auto tileM = Min(blockScheduler.mStop - coordM, blockScheduler.mTile);
            auto tensorBlockA =
                GetTile(tensorA, MakeCoord(coordM, _), MakeShape(blockScheduler.mStep, Get<1>(tensorA.GetShape())));
            auto tileN = blockScheduler.n0Tile;
            for (uint64_t coordN = blockScheduler.n0Start; coordN < blockScheduler.n0Stop;
                 coordN += blockScheduler.n0Step) {
                auto tensorBlockBias = GetBiasTile(tensorBias, coordN, tileN, params.mmad.isBias);
                auto tensorBlockC = GetTile(tensorC, MakeCoord(coordM, coordN), MakeShape(tileM, tileN));
                blockMmad(tensorBlockA, tensorBlockBias, tensorBlockC, MakeShape(tileM, tileN), params.mmad);
            }
            for (uint64_t coordN = blockScheduler.n1Start; coordN < blockScheduler.n1Stop;
                 coordN += blockScheduler.n1Step) {
                tileN = coordN + blockScheduler.n1Tile > blockScheduler.n1Stop ? blockScheduler.n1Stop - coordN :
                                                                                 blockScheduler.n1Tile;
                auto tensorBlockBias = GetBiasTile(tensorBias, coordN, tileN, params.mmad.isBias);
                auto tensorBlockC = GetTile(tensorC, MakeCoord(coordM, coordN), MakeShape(tileM, tileN));
                blockMmad(tensorBlockA, tensorBlockBias, tensorBlockC, MakeShape(tileM, tileN), params.mmad);
            }
            for (uint64_t coordN = blockScheduler.n2Start; coordN < blockScheduler.n2Stop;
                 coordN += blockScheduler.n2Step) {
                tileN = coordN + blockScheduler.n2Tile > blockScheduler.n2Stop ? blockScheduler.n2Stop - coordN :
                                                                                 blockScheduler.n2Tile;
                auto tensorBlockBias = GetBiasTile(tensorBias, coordN, tileN, params.mmad.isBias);
                auto tensorBlockC = GetTile(tensorC, MakeCoord(coordM, coordN), MakeShape(tileM, tileN));
                blockMmad(tensorBlockA, tensorBlockBias, tensorBlockC, MakeShape(tileM, tileN), params.mmad);
            }
        }
    }

private:
    static constexpr bool WEIGHT_NZ = IsZn2D<typename BlockPrologue::LayoutIn>::value;
    static constexpr QuantType ANTIQUANT_TYPE = QUANT_TYPE<
        AscendC::Std::remove_cvref_t<decltype(typename BlockPrologue::LayoutScale{}.GetShape())>,
        typename BlockPrologue::ElementScale>;

    template <typename ScaleOffsetType>
    __aicore__ inline auto GetAntiScaleTile(
        const ScaleOffsetType& scaleOffset, uint64_t coordN, uint64_t tileN, uint64_t antiQuantGroupSize)
    {
        if constexpr (ANTIQUANT_TYPE == QuantType::PER_TENSOR) {
            return scaleOffset;
        } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
            return GetTile(scaleOffset, MakeCoord(coordN, _), MakeShape(tileN, Cmct::Gemm::_1{}));
        } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_GROUP) {
            return GetTile(scaleOffset, MakeCoord(coordN, _), MakeShape(tileN, Get<1>(scaleOffset.GetShape())));
        } else if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
            return GetTile(scaleOffset, MakeCoord(coordN, _), MakeShape(tileN, Get<1>(scaleOffset.GetShape())));
        } else {
            static_assert(AscendC::Std::always_false_v<ScaleOffsetType>, "not support this ANTIQUANT_TYPE");
        }
    }

    template <bool IsPrivate, typename WeightType>
    __aicore__ inline auto GetWeightTile(
        const WeightType& weight, uint64_t coordN, uint64_t tileN, uint64_t antiQuantGroupSize)
    {
        if constexpr (IsPrivate) {
            return GetTile(
                weight, MakeCoord(MakeCoord(_, coordN / 16), MakeCoord(_, _)),
                MakeShape(MakeShape(Cmct::Gemm::_16{}, Cmct::CeilDiv(tileN, 16UL)), Get<1>(weight.GetShape())));
        } else {
            return GetTile(weight, MakeCoord(coordN, _), MakeShape(tileN, Get<1>(weight.GetShape())));
        }
    }

    template <typename BiasType>
    __aicore__ inline auto GetBiasTile(const BiasType& bias, uint64_t coordN, uint64_t tileN, bool hasBias)
    {
        if (hasBias) {
            return GetTile(bias, MakeCoord(coordN), MakeShape(tileN));
        } else {
            return bias;
        }
    }
};
} // namespace Cmct::Gemm::Kernel

#endif
