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
 * \file block_mmad_with_layout.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_WITH_LAYOUT_H
#define MATMUL_BLOCK_BLOCK_MMAD_WITH_LAYOUT_H

#include <type_traits>
#include "../utils/layout_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"
#include "./block_mmad.h"

namespace Cmct {
namespace Gemm {
namespace Block {
/**
* @class BlockMmadWithLayout
* @brief The intermediate of Block matrix multiplication class
*
* Inheriting from BlockMmadBase, encapsulates and adapts common functions for Layout representation
*/
template <
    class Derived,
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType,
    class TileCopy
>
class BlockMmadWithLayout
    : public BlockMmadBase<Derived, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy> {
public:
    using Base = BlockMmadBase<Derived, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    __aicore__ ~BlockMmadWithLayout()
    {
        End();
    }

    /**
    * @brief Initialize the matrix multiplication object
    */
    __aicore__ inline void Init()
    {
        this->AsDerived().matmul_.Init((TCubeTiling*)(nullptr), (AscendC::TPipe*)(nullptr));
    }

    /**
    * @brief IterateAll function to perform matrix multiplication iteration
    * @param [out] CTensor: destination tensor type
    * @param [in] ATensor: matrix A source tensor type
    * @param [in] BTensor: matrix B source tensor type
    * @param [in] Shape: shape type
    * @param [out] c: destination tensor
    * @param [in] a: matrix A source tensor
    * @param [in] b: matrix B source tensor
    * @param [in] actualShape: actual shape
    */
    template <class CTensor, class ATensor, class BTensor, class Shape>
    __aicore__ inline void IterateAll(CTensor& c, const ATensor& a, const BTensor& b, const Shape& actualShape)
    {
        // Convert a tensor with layout to a normal tensor
        auto aTensor = ConvertToTensorWithoutLayout<typename AType::T>(a);
        auto bTensor = ConvertToTensorWithoutLayout<typename BType::T>(b);
        auto cTensor = ConvertToTensorWithoutLayout<typename CType::T>(c);

        this->AsDerived().matmul_.SetTensorA(aTensor, AType::isTrans);
        this->AsDerived().matmul_.SetTensorB(bTensor, BType::isTrans);

        SetOrgShape(c, a, b);

        this->AsDerived().matmul_.SetSingleShape(Get<MNK_M>(actualShape), Get<MNK_N>(actualShape), Get<MNK_K>(actualShape));
        this->AsDerived().matmul_.IterateAll(cTensor);

        if constexpr (AscendC::is_global_tensor_v<CTensor>) {
            c.address_ = cTensor.address_;
        } else {
            c.SetAddr(cTensor.address_);
        }
    }

private:
    /**
    * @brief End the matrix multiplication operation
    */
    __aicore__ inline void End()
    {
        this->AsDerived().matmul_.End();
    }

    /**
    * @brief Set the original shape of the matrix
    *
    * Calculate and set the original shape of matrix c based on matrixs a and b
    *
    * @param [in] CTensor: the destination tensor type
    * @param [in] ATensor: the source tensor A type
    * @param [in] BTensor: the source tensor B type
    * @param [in] c: the destination tensor
    * @param [in] a: the source tensor A
    * @param [in] b: the source tensor B
    */
    template <class CTensor, class ATensor, class BTensor>
    __aicore__ inline void SetOrgShape(CTensor& c, const ATensor& a, const BTensor& b)
    {
        constexpr int32_t mIdx = AType::isTrans ? 1 : 0;
        constexpr int32_t nIdx = BType::isTrans ? 0 : 1;
        constexpr int32_t kaIdx = AType::isTrans ? 0 : 1;
        constexpr int32_t kbIdx = BType::isTrans ? 1 : 0;

        int32_t orgM;
        int32_t orgKa;
        const auto& aShape = a.GetTensorTrait().GetLayout().GetShape();
        if constexpr (IsNDOrAlign<AType>()) {
            orgM = Get<mIdx>(aShape);
            orgKa = Get<kaIdx>(aShape);
        } else if constexpr (IsNz<AType>()) {
            orgM = Get<mIdx, 0>(aShape) * Get<mIdx, 1>(aShape);
            orgKa = Get<kaIdx, 0>(aShape) * Get<kaIdx, 1>(aShape);
        } else {
            static_assert(AscendC::Std::always_false_v<AType>, "Format of A is not supported yet");
        }

        int32_t orgN;
        int32_t orgKb;
        const auto& bShape = b.GetTensorTrait().GetLayout().GetShape();
        if constexpr (IsNDOrAlign<BType>()) {
            orgN = Get<nIdx>(bShape);
            orgKb = Get<kbIdx>(bShape);
        } else if constexpr (IsNz<BType>()) {
            orgN = Get<nIdx, 0>(bShape) * Get<nIdx, 1>(bShape);
            orgKb = Get<kbIdx, 0>(bShape) * Get<kbIdx, 1>(bShape);
        } else {
            static_assert(AscendC::Std::always_false_v<BType>, "Format of B is not supported yet");
        }

        int32_t orgKc; // Set if matrix C's N != matrix B's N
        const auto& cShape = c.GetTensorTrait().GetLayout().GetShape();
        if constexpr (IsNDOrAlign<CType>()) {
            orgKc = Get<1>(cShape);
        } else if constexpr (IsNz<CType>()) {
            orgKc = Get<1, 0>(cShape) * Get<1, 1>(cShape);
        }
        this->AsDerived().matmul_.SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif