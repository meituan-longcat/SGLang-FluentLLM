/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*!
 * \file op_api_def.h
 * \brief
 */

#ifndef OP_API_DEF_H_
#define OP_API_DEF_H_

namespace op {
    const size_t BN_MIN_SUPPORT_DIMS_NUMS = 2;
    const size_t MAX_SUPPORT_DIMS_NUMS = 8;
    const int8_t FP16FP32_KEEP_DTYPE = -1;
    const int8_t KEEP_DTYPE = 0;
    const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
    const int8_t USE_FP16 = 2;
    const int8_t USE_HF32 = 3;
    constexpr size_t MAX_MASK_LEN64 = 64;
}  // namespace op
#endif  // OP_API_DEF_H_