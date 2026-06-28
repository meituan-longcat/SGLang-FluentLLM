/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
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
 * \file runtime2_util.h
 * \brief runtime2 util
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_

#include <nlohmann/json.hpp>
#include "register/op_impl_registry.h"
#include "runtime/tiling_parse_context.h"
#include "context_util.h"
#include "op_util.h"
#include "../error_log.h"
#include "../op_tiling_util.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "op_tiling/cube/algorithm/hash/hash.h"
#include "graph/ge_error_codes.h"
#include "platform/platform_infos_def.h"
#include "shape_utils.h"
#include "getcdim.h"

namespace optiling {
template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}

/*
 * @brief: get Byte size base on dtype(string)
 * @param [in] op_type: string dtype
 * @return DataType: ge dtype
 */
inline ge::DataType GetGeTypeFromStr(const std::string& dtype_str) {
  const std::map<std::string, ge::DataType> str_2_datatype = {
    {"float", ge::DT_FLOAT},
    {"float32", ge::DT_FLOAT},
    {"float16", ge::DT_FLOAT16},
    {"int8", ge::DT_INT8},
    {"int16", ge::DT_INT16},
    {"int32", ge::DT_INT32},
    {"int64", ge::DT_INT64},
    {"uint8", ge::DT_UINT8},
    {"uint16", ge::DT_UINT16},
    {"uint32", ge::DT_UINT32},
    {"uint64", ge::DT_UINT64},
    {"bool", ge::DT_BOOL},
    {"double", ge::DT_DOUBLE},
    {"dual", ge::DT_DUAL},
    {"dual_sub_int8", ge::DT_DUAL_SUB_INT8},
    {"dual_sub_uint8", ge::DT_DUAL_SUB_UINT8},
    {"int4", ge::DT_INT4},
    {"bfloat16", ge::DT_BF16}};

  auto find_it = str_2_datatype.find(dtype_str);
  if (find_it != str_2_datatype.end()) {
    return find_it->second;
  }
  OP_LOGW("GetGeTypeFromStr", "con not get the dtype[%s] in ge::DataType list. will return DT_MAX", dtype_str.c_str());
  return ge::DT_MAX;
}

/*
 * @brief: Calculate reduce cof value
 * @param [in] input_shape: gert::Shape, the input shape for reduce
 * @param [in] reduce_axis: const std::vector<int32_t>, the reduce axes num
 * @param [out] reduce_mean_cof: the result of reduce cof value
 * @return bool: true or false;
 */
template <typename T>
bool CalcReduceMeanCof(const gert::Shape& input_shape, const std::vector<T>& reduce_axis,
                       float& reduce_mean_cof) {
  const size_t dim_len = input_shape.GetDimNum();
  const size_t ori_reduce_axis_len = reduce_axis.size();
  // init reduce_mean_cof is 1.0
  reduce_mean_cof = 1.0;
  for (size_t i = 0; i < ori_reduce_axis_len; i++) {
    OP_TILING_CHECK(
        !ops::IsDimValid(dim_len, reduce_axis[i]),
        VECTOR_INNER_ERR_REPORT_TILIING("CalcReduceMeanCof", "%s",
                                        ops::GenInvalidDimMsg("reduce_axis", i, dim_len, reduce_axis[i]).c_str()),
        return false);

    // convert reduce axis (like: -1 -> (dim_len - 1))
    T single_reduce_axis = reduce_axis[i] < 0 ? reduce_axis[i] + dim_len : reduce_axis[i];

    int64_t reduce_dim = input_shape.GetDim(single_reduce_axis);
    OP_TILING_CHECK(reduce_dim == 0, OP_LOGI("CalcReduceMeanCof", "the reduce dim is 0, will ignore reduce_mean_cof"),
                    return true);
    reduce_mean_cof = reduce_mean_cof / reduce_dim;
  }
  OP_LOGD("CalcReduceMeanCof", "CalcReduceMeanCof cof is %1f", reduce_mean_cof);

  return true;
}

/*
 * @brief: add reduce cof value after the tiling data
 * @param [in] input_shape: gert::Shape, the input shape for reduce
 * @param [in] input_dtype: ge::DataType,  the input dtype for reduce
 * @param [in] reduce_axis: const std::vector<int32_t>, the reduce axes num
 * @param [out] tiling_data: gert::TilingData, the tiling data, will add the cof value to th lasy TilingData
 * @return bool: true or false;
 */
template <typename T>
bool AddReduceMeanCof(const gert::Shape& input_shape, const ge::DataType input_dtype,
                      const std::vector<T>& reduce_axis, gert::TilingData* tiling_data) {
  float reduce_mean_cof = 1.0;
  bool calcu_flag = CalcReduceMeanCof(input_shape, reduce_axis, reduce_mean_cof);
  OP_LOGD("AddReduceMeanCof", "AddReduceMeanCof dtype is %s", ops::ToString(input_dtype).c_str());
  switch (input_dtype) {
    case ge::DT_FLOAT:
      tiling_data->Append((float)reduce_mean_cof);
      return calcu_flag;
    case ge::DT_FLOAT16:
      tiling_data->Append((fe::fp16_t)reduce_mean_cof);
      tiling_data->Append((uint16_t)0);
      return calcu_flag;
    default:
      OP_LOGW("AddReduceMeanCof", "Only support [DT_FLOAT, DT_FLOAT16], but is [%s]",
              ops::ToString(input_dtype).c_str());
      return false;
  }
}

/*
 * @brief: get the json class of compile info from context
 * @param [in] context: gert::TilingContext
 * @return bool: std::unique_ptr<nlohmann::json>;
 */
inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context) {
  auto json_str = context->GetCompiledJson();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, json_str, nullptr);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  return parsed_object_cinfo;
}

/*
 * @brief: get the json class of compile info from context
 * @param [in] context: gert::TilingContext
 * @param [out] ci_key: uint32_t
 * @return bool: std::unique_ptr<nlohmann::json>;
 */
inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context, uint32_t& ci_key) {
  auto json_str = context->GetCompiledJson();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, json_str, nullptr);
  ci_key = cachetiling::MurmurHash(json_str, std::strlen(json_str));
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  return parsed_object_cinfo;
}

/*
 * @brief: read one compile item from json object
 * @param [in] all_vars: json object
 * @param [in] name: item name to read
 * @param [out] value: item value to load
 * @return bool: success or failed
 */
template <typename T>
bool ReadCompileItem(const nlohmann::json& all_vars, const std::string& name, T& value) {
  if (all_vars.empty()) {
    return false;
  }

  if (all_vars.count(name) == 0) {
    return false;
  }

  value = all_vars[name].get<T>();
  return true;
}

template <typename T1, typename T2>
void ReadCompileItem(const nlohmann::json& all_vars, const std::string& name, T1& value, const T2 default_value) {
  if (!ReadCompileItem(all_vars, name, value)) {
    value = static_cast<T1>(default_value);
  }
}

/*
 * @brief: add workspace size to context
 * @param [in] context: gert::TilingContext
 * @param [in] workspace: size_t the workspace num
 * @return bool: true or false;
 */
bool AddWorkspace(gert::TilingContext* context, const size_t workspace);

/*
 * @brief: ceil(u_value/d_value)*d_value
 *         eg. CeilAlign(4,3) -> 6, CeilAlign(4,2) -> 4, CeilAlign(4,0) -> 4
 * @param [in] u_value: int64_t
 * @param [in] d_value: int64_t
 * @return int64: ceil
 */
int64_t CeilAlign(int64_t u_value, int64_t d_value);

/*
 * @brief: if d_value == 0 return u_value, else return u_value % d_value
 * @param [in] u_value: int64_t
 * @param [in] d_value: int64_t
 * @return int64: ceil
 */
int64_t GetRemainder(int64_t u_value, int64_t d_value);

/*
 * @brief: calculate the shape size of shape from begin to end
 * @param [in] shape: gert::Shape, the input shape
 * @param [in] begin: size_t, the begin point
 * @param [in] end: size_t, the end point
 * @return int64: the total shape size for begin to end
 */
int64_t GetPartShapeSize(const gert::Shape& shape, size_t begin, size_t end);

/*
 * @brief: for debug print runtime2 tiling data
 * @param [in] context: gert::TilingContext
 * @param [out] string: tiling data string
 * @return string: result
 */
template <typename T>
std::string GetTilingDataString(gert::TilingContext* context) {
  auto tiling_data = context->GetRawTilingData();
  auto data_size = tiling_data->GetDataSize();
  std::string result;
  const T *data = reinterpret_cast<const T*>(tiling_data->GetData());
  size_t len = data_size / sizeof(T);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

template <typename T>
void AddTailCleanVar(gert::TilingContext* context, const size_t index, gert::TilingData* tiling_data) {
  int64_t tail_clean_offset = ops::GetInputCDim(context, index);
  int64_t tail_clean_align = CeilAlign(tail_clean_offset, 16);
  tiling_data->Append(static_cast<T>(tail_clean_offset));
  tiling_data->Append(static_cast<T>(tail_clean_align - tail_clean_offset));
}

/*
 * @brief: get running core number.
 * @param [in] TilingParseContext: context
 * @param [out] int64_t: running core number
 * @return bool: result
 */
bool GetTilingCoreNum(const gert::TilingParseContext* context, uint32_t& core_num);

bool IsRegbaseSocVersion(const gert::TilingParseContext& context);
bool IsRegbaseSocVersion(const gert::TilingContext& context);

template <typename T>
class OpHashInput {
public:
  OpHashInput(const OpHashInput& rhs) {
    content = rhs.content;
  }
  OpHashInput() {
  }

  explicit OpHashInput(T& input) {
    content = input;
  }

  OpHashInput& operator=(const OpHashInput& rhs) {
    content = rhs.content;
    return *this;
  }

  bool operator==(const OpHashInput& input) const {
    if (std::memcmp(&content, &input.content, sizeof(content)) == 0) {
      return true;
    }
    return false;
  }

private:
  T content;
};
}  // namespace optiling
#endif  // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_