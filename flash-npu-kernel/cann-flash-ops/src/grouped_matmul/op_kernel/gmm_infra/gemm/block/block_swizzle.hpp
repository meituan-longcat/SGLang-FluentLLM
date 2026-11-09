/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP

#include "../../../gmm_infra/base_defs.hpp"
#include "../../../gmm_infra/detail/alignment.hpp"
#include "../../../gmm_infra/gemm_coord.hpp"
#include "../../../gmm_infra/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Block swizzling function for Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    /// Data members

    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    /// Methods

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle()
    {
    }

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_)
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_)
    {
    }

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;

        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = loopsMN_;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / (GetCoreLoops());
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMN.column() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mActual =
            (blockCoord.m() == (loopsMN.row() - 1)) ? (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
                               (problemShape.n() - blockCoord.n() * tileMN.column()) :
                               tileMN.column();
        uint32_t kActual = problemShape.k();
        return GemmCoord{mActual, nActual, kActual};
    }
};

/// Block swizzling function for Splitk Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SplitkGemmIdentityBlockSwizzle {
    /// Data members
    GemmCoord problemShape;
    GemmCoord tileShape;
    GemmCoord loopsMNK;
    uint32_t splitkFactor = 1; // splite k dim into virtual cores


    /// Methods

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle()
    {
    }

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileShape_,
                                   uint32_t splitkFactor_ = 1)
        : problemShape(problemShape_), tileShape(tileShape_), splitkFactor(splitkFactor_)
    {
        loopsMNK = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    uint32_t GetKIdxBySplitkSliceIdx(uint32_t splitkSliceIdx) const
    {
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            return (loopsMNK.k() / splitkFactor + 1) * splitkSliceIdx;
        } else {
            return splitkSliceIdx * (loopsMNK.k() / splitkFactor) + loopsMNK.k() % splitkFactor;
        }
    }

    CATLASS_DEVICE
    uint32_t GetSplitkSliceIdx(uint32_t taskIdx) const
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        return taskIdx % GetCoreLoops() / mnLoops;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMNK.m() * loopsMNK.n() * splitkFactor;
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / GetCoreLoops();
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord, uint32_t splitkSliceIdx)
    {
        uint32_t splitkSliceLen;
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            splitkSliceLen = (loopsMNK.k() / splitkFactor + 1) * tileShape.k();
        } else {
            splitkSliceLen = (loopsMNK.k() / splitkFactor) * tileShape.k();
        }
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
                               (problemShape.m() - blockCoord.m() * tileShape.m()) :
                               tileShape.m();
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
                               (problemShape.n() - blockCoord.n() * tileShape.n()) :
                               tileShape.n();
        uint32_t kActual = (splitkSliceIdx == (splitkFactor - 1)) ?
                               (problemShape.k() - blockCoord.k() * tileShape.k()) :
                               splitkSliceLen;
        return GemmCoord{mActual, nActual, kActual};
    }
};

/// Block swizzling function for Splitk Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SplitkInOneCoreGemmIdentityBlockSwizzle {
    /// Data members

    static constexpr uint32_t MAX_TILE_NUM = 5000;
    static constexpr uint32_t MAX_LINE = 500;

    GemmCoord problemShape;
    GemmCoord tileShape;
    GemmCoord loopsMNK;
    uint32_t splitkFactor = 1;
    uint32_t maxKValue = 0;
    uint32_t kPerLoop = 0;
    uint32_t blockCntPerCore = 0;
    uint32_t leftBlockCnt = 0;
    uint32_t minPlusBlockIdx = 0;
    uint32_t maxPlusBlockIdx = 0;
    uint32_t blockIdexList[MAX_TILE_NUM];
    uint32_t blockCnt = 0;
    uint32_t beginBlockIdex = 0;

    /// Methods
    CATLASS_DEVICE
    SplitkInOneCoreGemmIdentityBlockSwizzle()
    {
    }

    CATLASS_DEVICE
    SplitkInOneCoreGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileShape_,
                                            uint32_t splitkFactor_ = 1)
        : problemShape(problemShape_), tileShape(tileShape_), splitkFactor(splitkFactor_)
    {
        loopsMNK = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    uint32_t GetKIdxBySplitkSliceIdx(uint32_t splitkSliceIdx) const
    {
        return kPerLoop * splitkSliceIdx;
    }

    CATLASS_DEVICE
    uint32_t GetSplitkSliceIdx(uint32_t taskIdx) const
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        return taskIdx % splitkFactor;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops(uint32_t coreIdx) const
    {
        uint32_t blockCntTmp = blockCntPerCore;
        if (leftBlockCnt > 0 &&
            ((minPlusBlockIdx <= maxPlusBlockIdx && coreIdx >= minPlusBlockIdx && coreIdx <= maxPlusBlockIdx) ||
             (minPlusBlockIdx > maxPlusBlockIdx && (coreIdx >= minPlusBlockIdx || coreIdx <= maxPlusBlockIdx)))) {
            blockCntTmp = blockCntPerCore + 1;
        }
        return blockCntTmp;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return blockCnt;
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx % splitkFactor;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx / splitkFactor % (loopsMNK.m() * loopsMNK.n());
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord, uint32_t splitkSliceIdx)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
                               (problemShape.m() - blockCoord.m() * tileShape.m()) :
                               tileShape.m();
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
                               (problemShape.n() - blockCoord.n() * tileShape.n()) :
                               tileShape.n();
        uint32_t kActual =
            (splitkSliceIdx == (splitkFactor - 1)) ? (problemShape.k() - blockCoord.k() * tileShape.k()) : maxKValue;
        return GemmCoord{mActual, nActual, kActual};
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockTailBefore(GemmCoord blockCoord, uint32_t splitkSliceIdx)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
                               (problemShape.m() - blockCoord.m() * tileShape.m()) :
                               tileShape.m();
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
                               (problemShape.n() - blockCoord.n() * tileShape.n()) :
                               tileShape.n();
        uint32_t kActual;
        if (splitkFactor > 1) {
            kActual = (splitkSliceIdx == (splitkFactor - 2)) ?
                          (problemShape.k() - blockCoord.k() * tileShape.k() - maxKValue) :
                          maxKValue;
        } else {
            kActual = (splitkSliceIdx == (splitkFactor - 1)) ? (problemShape.k() - blockCoord.k() * tileShape.k()) :
                                                               maxKValue;
        }

        return GemmCoord{mActual, nActual, kActual};
    }

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, GemmCoord const &tileShape_, uint32_t splitkFactor_,
                uint32_t maxKValue_)
    {
        problemShape = problemShape_;
        tileShape = tileShape_;
        splitkFactor = splitkFactor_;
        maxKValue = maxKValue_;

        loopsMNK = CeilDiv(problemShape, tileShape);
        kPerLoop = maxKValue / tileShape.k();
    }

    CATLASS_DEVICE
    void SetBlockIdx(uint32_t coreNum, uint32_t coreIdx)
    {
        uint32_t blockDimM = loopsMNK.m();
        uint32_t blockDimN = loopsMNK.n();
        uint32_t lastNIdx[MAX_LINE] /*= {0}*/;
        for (uint32_t i = 0; i < blockDimM; i++) {
            lastNIdx[i] = 0;
        }
        for (uint32_t i = 0; i <= coreIdx; i++) {
            uint32_t line = i % blockDimM;
            uint32_t col = i / blockDimM;
            if (col % 2 != 0) {
                line = (col + 1) * blockDimM - i - 1;
            }
            uint32_t offset = coreIdx / 4;
            uint32_t blockCntTmp = blockCnt;
            if (i != coreIdx) {
                blockCntTmp = GetCoreLoops(i);
            }
            for (uint32_t j = 0; j < blockCntTmp; j++) {
                if (col % 2 == 0) {
                    while (lastNIdx[line] >= blockDimN) {
                        line++;
                    }
                } else {
                    while (lastNIdx[line] >= blockDimN) {
                        line--;
                    }
                }
                if (i == coreIdx) {
                    uint32_t jOffset = (j + offset) % blockCnt;
                    blockIdexList[jOffset] = lastNIdx[line] + line * blockDimN;
                }
                lastNIdx[line]++;
            }
        }
    }

    CATLASS_DEVICE
    void SetBeginBlockIdx(uint32_t coreNum, uint32_t coreIdx)
    {
        for (uint32_t i = 0; i < coreIdx; i++) {
            beginBlockIdex += GetCoreLoops(i);
        }
    }

    CATLASS_DEVICE
    uint32_t GetCoreBlockIdx(uint32_t index)
    {
        return blockIdexList[index];
    }

    CATLASS_DEVICE
    uint32_t GetCoreBeginBlockIdx()
    {
        return beginBlockIdex;
    }

    CATLASS_DEVICE
    uint8_t GetNeedLeftCopyFlag(uint32_t index)
    {
        uint32_t curBlockLine = 0;
        uint32_t blockDimN = loopsMNK.n();
        if (index == blockCnt - 1) {
            return 1;
        }
        if (index > 1) {
            uint32_t lastBlockLine = blockIdexList[index - 1] / blockDimN;
            uint32_t curBlockLine = blockIdexList[index] / blockDimN;
            if (lastBlockLine != curBlockLine) {
                return 1;
            }
        }
        return 0;
    }

    CATLASS_DEVICE
    uint8_t GetNeedLeftCopyFlag(uint32_t index, uint32_t curBlockIdex)
    {
        uint32_t curBlockLine = 0;
        uint32_t blockDimN = loopsMNK.n();
        if (index == blockCnt - 1) {
            return 1;
        }
        if (index > 1) {
            uint32_t lastBlockLine = (curBlockIdex - 1) / blockDimN;
            uint32_t curBlockLine = curBlockIdex / blockDimN;
            if (lastBlockLine != curBlockLine) {
                return 1;
            }
        }
        return 0;
    }

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, GemmCoord const &tileShape_, uint32_t splitkFactor_,
                uint32_t maxKValue_, uint32_t groupIdx, uint32_t coreNum, uint32_t coreIdx)
    {
        problemShape = problemShape_;
        tileShape = tileShape_;
        splitkFactor = splitkFactor_;
        maxKValue = maxKValue_;

        loopsMNK = CeilDiv(problemShape, tileShape);
        kPerLoop = maxKValue / tileShape.k();

        uint32_t totalBlockCnt = loopsMNK.m() * loopsMNK.n();
        blockCntPerCore = totalBlockCnt / coreNum;
        leftBlockCnt = totalBlockCnt - blockCntPerCore * coreNum;
        minPlusBlockIdx = (groupIdx * leftBlockCnt) % coreNum;
        maxPlusBlockIdx = (groupIdx * leftBlockCnt + leftBlockCnt - 1) % coreNum;

        blockCnt = GetCoreLoops(coreIdx);
        SetBlockIdx(coreNum, coreIdx);
    }

    CATLASS_DEVICE
    void UpdateEx(GemmCoord const &problemShape_, GemmCoord const &tileShape_, uint32_t splitkFactor_,
                  uint32_t maxKValue_, uint32_t groupIdx, uint32_t coreNum, uint32_t coreIdx)
    {
        problemShape = problemShape_;
        tileShape = tileShape_;
        splitkFactor = splitkFactor_;
        maxKValue = maxKValue_;

        loopsMNK = CeilDiv(problemShape, tileShape);
        kPerLoop = maxKValue / tileShape.k();

        uint32_t totalBlockCnt = loopsMNK.m() * loopsMNK.n();
        blockCntPerCore = totalBlockCnt / coreNum;
        leftBlockCnt = totalBlockCnt - blockCntPerCore * coreNum;
        minPlusBlockIdx = (groupIdx * leftBlockCnt) % coreNum;
        maxPlusBlockIdx = (groupIdx * leftBlockCnt + leftBlockCnt - 1) % coreNum;

        blockCnt = GetCoreLoops(coreIdx);
        SetBeginBlockIdx(coreNum, coreIdx);
    }
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP
