/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file example.h
 * \brief
 */
#ifndef GE_OP_ZEROSLIKE_H
#define GE_OP_ZEROSLIKE_H

#include "../graph/operator_reg.h"
namespace ge {

REG_OP(Zeroslike)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(dst_dtype, String, "")
    .ATTR(optimize, Bool, true) // only surport true
    .INFER_SHAPE_AND_TYPE(ELMTWISE_INFER("x", "y"))
    .OP_END_FACTORY_REG(Zeroslike)
} // namespace ge

#endif // GE_OP_REDUCE_ZEROSLIKE_H
