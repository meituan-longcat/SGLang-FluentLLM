/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "fallback_comm.h"

#include <iostream>
#include <unordered_map>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "runtime/base.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace std;
using namespace gert;
using namespace ge;

aclDataType ToAclDataType(ge::DataType dtype) {
  static const std::vector<DataType> CANN_CONVERT_TO_ACL_DataType_LIST = {
      ge::DataType::DT_FLOAT,     ge::DataType::DT_FLOAT16,    ge::DataType::DT_INT8,   ge::DataType::DT_INT32,
      ge::DataType::DT_UINT8,     ge::DataType::DT_INT16,      ge::DataType::DT_UINT16, ge::DataType::DT_UINT32,
      ge::DataType::DT_INT64,     ge::DataType::DT_DOUBLE,     ge::DataType::DT_BOOL,   ge::DataType::DT_STRING,
      ge::DataType::DT_COMPLEX64, ge::DataType::DT_COMPLEX128, ge::DataType::DT_BF16,  ge::DataType::DT_UINT64,
      ge::DataType::DT_INT4};
  auto iter = std::find(CANN_CONVERT_TO_ACL_DataType_LIST.begin(), CANN_CONVERT_TO_ACL_DataType_LIST.end(), dtype);
  if (iter == CANN_CONVERT_TO_ACL_DataType_LIST.end()) {
    return aclDataType::ACL_DT_UNDEFINED;
  }
  return static_cast<aclDataType>(dtype);
}

}  // namespace fallback

#ifdef __cplusplus
}
#endif
