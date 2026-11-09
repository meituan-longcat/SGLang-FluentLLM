/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_v3_scatter_cache.h
 * \brief
 */

#ifndef MLA_PROLOG_V3_SCATTER_CACHE_H
#define MLA_PROLOG_V3_SCATTER_CACHE_H

#include "mla_prolog_v3_comm.h"

namespace MlaPrologV3 {

__aicore__ inline int64_t CeilDiv64(int64_t x, int64_t y)
{
    if (y == 0) {
        return static_cast<int64_t>(0);
    }
    return (x + y - 1) / y;
};

/**
 * @brief PA场景，将inputLocal中的数据scatter到cacheGm，支持ND和Nz cache
 * @param cacheGm 输出tensor
 *     ND [blockNum, blockSize, col]
 *     Nz [blockNum, ceil(col/col0), blockSize, col0]
 * @param inputLocal 输入tensor，[row, col]，一行对应一个token，只支持单行数据处理，row为1
 * @param scatterCacheParams 所需相关参数，包括
          blockSize KV blocks的大小
          paTokenIndex 待处理token在cache中的全局index，取值[0, blockNum*blockSize)
          row 待处理的行数
          col 待处理的列数，需满足32 bytes对齐
 */

struct ScatterCacheParams {
    int64_t blockSize;
    int64_t paTokenIndex;
    int64_t row;
    int64_t col;
    int64_t stride;
    int64_t seqLength;
    int64_t tokenIndex;
};

template <typename T, bool isNz>
__aicore__ inline void ScatterCache(const GlobalTensor<T>& cacheGm, const LocalTensor<T>& inputLocal,
                                    const ScatterCacheParams& scatterCacheParams) {
    if (scatterCacheParams.paTokenIndex < 0) {
        return;
    }
    if constexpr (!isNz) {
        DataCopy(cacheGm[scatterCacheParams.paTokenIndex * scatterCacheParams.stride],
                inputLocal, scatterCacheParams.col);
    } else {
        constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        int64_t cacheOffset = scatterCacheParams.paTokenIndex / scatterCacheParams.blockSize *
                            scatterCacheParams.blockSize * scatterCacheParams.stride +
                            scatterCacheParams.paTokenIndex % scatterCacheParams.blockSize * col0;
        DataCopyParams copyParams {static_cast<uint16_t>(scatterCacheParams.col / col0),
                                1, 0, static_cast<uint16_t>(scatterCacheParams.blockSize - 1)};
        DataCopy(cacheGm[cacheOffset], inputLocal, copyParams);
    }
}

template <typename T, bool isNz>
__aicore__ inline void ScatterCacheUnAligned(const GlobalTensor<T>& cacheGm, const LocalTensor<T>& inputLocal,
                                             const ScatterCacheParams& scatterCacheParams) {
    if (scatterCacheParams.paTokenIndex < 0) {
        return;
    }
    if constexpr (!isNz) {
        // blockCount, blockLen, srcStride, dstStride
        DataCopyParams dataCopyParams {1, static_cast<uint16_t>(scatterCacheParams.col * sizeof(T)), 0, 0};
        DataCopyPad(cacheGm[scatterCacheParams.paTokenIndex * scatterCacheParams.stride], inputLocal, dataCopyParams);
    }
}

template <typename T, bool isNz>
__aicore__ inline void ScatterCacheMultiRows(GlobalTensor<T>& cacheGm, const LocalTensor<T>& inputLocal,
                                             const ScatterCacheParams& scatterCacheParams, int64_t rowsInCurBatch,
                                             int64_t cacheOffset, int64_t nextBatchOffset) {
    if (cacheOffset < 0) {
        return;
    }
    int64_t copyCnt = scatterCacheParams.col * rowsInCurBatch;

    if constexpr (!isNz) {
        DataCopy(cacheGm[cacheOffset], inputLocal, copyCnt);
        if (rowsInCurBatch != scatterCacheParams.row) {
            DataCopy(cacheGm[nextBatchOffset],
                    inputLocal[copyCnt], (scatterCacheParams.row - rowsInCurBatch)*scatterCacheParams.col);
        }
    } else {
        constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        DataCopyParams copyParams {static_cast<uint16_t>(copyCnt / col0),
                                1, 0, static_cast<uint16_t>(scatterCacheParams.blockSize - 1)};
        DataCopy(cacheGm[cacheOffset], inputLocal, copyParams);
        if (rowsInCurBatch != scatterCacheParams.row) {
            copyParams.blockCount = (scatterCacheParams.row - rowsInCurBatch)*scatterCacheParams.col / col0;
            DataCopy(cacheGm[nextBatchOffset], inputLocal[copyCnt],
                     (scatterCacheParams.row - rowsInCurBatch) * scatterCacheParams.col);
        }
    }
}

}

#endif