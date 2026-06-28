/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file op_tiling_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_

#include <vector>
#include <nlohmann/json.hpp>
#include "error_log.h"
#include "op_tiling.h"
#include "op_attr.h"
#include "op_const.h"
#include "op_util.h"
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "vector_tiling_profiling.h"

#define REGISTER_OP_TILING_V3_WITH_VECTOR(optype, opfunc, vector_key, optional_key)                                \
  bool Tbe##optype##TilingV3WithVec(const ge::Operator& para, const void* op_info_void,                            \
                                    optiling::utils::OpRunInfo& rinfo) {                                           \
    OP_TILING_CHECK(op_info_void == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(#optype, "op_info_void is nullptr."), \
                    return false);                                                                                 \
    return opfunc(#optype, para, *(const std::vector<int64_t>*)op_info_void, rinfo);                               \
  }                                                                                                                \
  void* Tbe##optype##TilingV3WithVecParsefunc(const ge::Operator& para, const ge::AscendString& compile_info) {    \
    return ParseCompileToInt64Vec(para, compile_info, vector_key, optional_key);                                   \
  }                                                                                                                \
  REGISTER_OP_TILING_V3(optype, Tbe##optype##TilingV3WithVec, Tbe##optype##TilingV3WithVecParsefunc)

#define REGISTER_OP_TILING_V3_CUSTOM(optype, opfunc, parse_func, struct_name)                                         \
  bool Tbe##optype##TilingV3Custom(const ge::Operator& para, const void* op_info_void,                                \
                                   optiling::utils::OpRunInfo& rinfo) {                                               \
    OP_TILING_CHECK(op_info_void == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(#optype, "op_info_void is nullptr."),    \
                    return false);                                                                                    \
    return opfunc(#optype, para, *static_cast<const struct_name*>(op_info_void), rinfo);                              \
  }                                                                                                                   \
  void* Tbe##optype##TilingV3CustomParsefunc(const ge::Operator& para, const ge::AscendString& compile_info) {        \
    std::shared_ptr<nlohmann::json> json_object(new nlohmann::json(nlohmann::json::parse(compile_info.GetString()))); \
    if (json_object == nullptr) {                                                                                     \
      return nullptr;                                                                                                 \
    }                                                                                                                 \
    struct_name* parsed_void_ptr = new (struct_name)();                                                               \
    bool parse_ret = parse_func(#optype, *json_object, *parsed_void_ptr);                                             \
    if (parse_ret) {                                                                                                  \
      return static_cast<void*>(parsed_void_ptr);                                                                     \
    }                                                                                                                 \
    delete parsed_void_ptr;                                                                                           \
    return nullptr;                                                                                                   \
  }                                                                                                                   \
  REGISTER_OP_TILING_V3(optype, Tbe##optype##TilingV3Custom, Tbe##optype##TilingV3CustomParsefunc)

#define REGISTER_OP_TILING_V4_WITH_VECTOR(optype, opfunc, vector_key, optional_key)                                  \
  class Tbe##optype##VecCompileInfo : public CompileInfoBase {                                                       \
   public:                                                                                                           \
    ~Tbe##optype##VecCompileInfo() = default;                                                                        \
    std::vector<int64_t> compile_vec;                                                                                \
  };                                                                                                                 \
  bool Tbe##optype##TilingV4WithVec(const ge::Operator& para, const std::shared_ptr<CompileInfoBase> op_info_ptr,    \
                                    optiling::utils::OpRunInfo& rinfo) {                                             \
    OP_TILING_CHECK(op_info_ptr == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(#optype, "op_info_ptr is nullptr."),     \
                    return false);                                                                                   \
    const std::shared_ptr<Tbe##optype##VecCompileInfo> compile_ptr =                                                 \
        std::dynamic_pointer_cast<Tbe##optype##VecCompileInfo>(op_info_ptr);                                         \
    OP_TILING_CHECK(compile_ptr == nullptr,                                                                          \
                    VECTOR_INNER_ERR_REPORT_TILIING(#optype, "change CompileInfoBase to VecCompileInfo failed."),    \
                    return false);                                                                                   \
    return opfunc(#optype, para, compile_ptr->compile_vec, rinfo);                                                   \
  }                                                                                                                  \
  std::shared_ptr<CompileInfoBase> Tbe##optype##TilingV4WithVecParsefunc(const ge::Operator& para,                   \
                                                                         const ge::AscendString& compile_info) {     \
    std::shared_ptr<Tbe##optype##VecCompileInfo> compile_ptr =                                                       \
        ops::make_shared_nothrow<Tbe##optype##VecCompileInfo>();                                                     \
    if (compile_ptr == nullptr) {                                                                                    \
      OP_LOGW(#optype, "make_shared failed, will return nullptr!");                                                  \
      return nullptr;                                                                                                \
    }                                                                                                                \
    bool parse_ret = ParseCompileToInt64Vec(para, compile_info, vector_key, optional_key, compile_ptr->compile_vec); \
    if (parse_ret) {                                                                                                 \
      return compile_ptr;                                                                                            \
    }                                                                                                                \
    return nullptr;                                                                                                  \
  }                                                                                                                  \
  REGISTER_OP_TILING_V4(optype, Tbe##optype##TilingV4WithVec, Tbe##optype##TilingV4WithVecParsefunc)

#define REGISTER_OP_TILING_V4_CUSTOM(optype, opfunc, parse_func, struct_name)                                        \
  class Tbe##optype##CustomCompileInfo : public CompileInfoBase {                                                    \
   public:                                                                                                           \
    ~Tbe##optype##CustomCompileInfo() = default;                                                                     \
    struct_name compile_info;                                                                                        \
  };                                                                                                                 \
  bool Tbe##optype##TilingV4Custom(const ge::Operator& para, const std::shared_ptr<CompileInfoBase> op_info_ptr,     \
                                   optiling::utils::OpRunInfo& rinfo) {                                              \
    OP_TILING_CHECK(op_info_ptr == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(#optype, "op_info_ptr is nullptr."),     \
                    return false);                                                                                   \
    const std::shared_ptr<Tbe##optype##CustomCompileInfo> compile_ptr =                                              \
        std::dynamic_pointer_cast<Tbe##optype##CustomCompileInfo>(op_info_ptr);                                      \
    OP_TILING_CHECK(compile_ptr == nullptr,                                                                          \
                    VECTOR_INNER_ERR_REPORT_TILIING(#optype, "change CompileInfoBase to CustomCompileInfo failed."), \
                    return false);                                                                                   \
    return opfunc(#optype, para, compile_ptr->compile_info, rinfo);                                                  \
  }                                                                                                                  \
  std::shared_ptr<CompileInfoBase> Tbe##optype##TilingV4CustomParsefunc(const ge::Operator& para,                    \
                                                                        const ge::AscendString& compile_info) {      \
    std::shared_ptr<nlohmann::json> json_object =                                                                    \
        ops::make_shared_nothrow<nlohmann::json>(nlohmann::json::parse(compile_info.GetString()));                   \
    if (json_object == nullptr) {                                                                                    \
      OP_LOGW(#optype, "nlohmann::json::parse the compile info failed, will return nullptr!");                       \
      return nullptr;                                                                                                \
    }                                                                                                                \
    std::shared_ptr<Tbe##optype##CustomCompileInfo> compile_ptr =                                                    \
        ops::make_shared_nothrow<Tbe##optype##CustomCompileInfo>();                                                  \
    if (compile_ptr == nullptr) {                                                                                    \
      OP_LOGW(#optype, "make_shared failed, will return nullptr!");                                                  \
      return nullptr;                                                                                                \
    }                                                                                                                \
    bool parse_ret = parse_func(#optype, *json_object, compile_ptr->compile_info);                                   \
    if (parse_ret) {                                                                                                 \
      return compile_ptr;                                                                                            \
    }                                                                                                                \
    OP_LOGW(#optype, "do parse_func failed, will return nullptr!");                                                  \
    return nullptr;                                                                                                  \
  }                                                                                                                  \
  REGISTER_OP_TILING_V4(optype, Tbe##optype##TilingV4Custom, Tbe##optype##TilingV4CustomParsefunc)

#define REGISTER_OP_TILING_V4_CUBE_PATTERN(oppattern, opfunc, parse_func, struct_name)                              \
  class Tbe##oppattern##CustomCompileInfo : public CompileInfoBase {                                                \
   public:                                                                                                          \
    ~Tbe##oppattern##CustomCompileInfo() = default;                                                                 \
    struct_name compile_info;                                                                                       \
  };                                                                                                                \
  bool Tbe##oppattern##TilingV4Custom(const ge::Operator &para, const std::shared_ptr<CompileInfoBase> op_info_ptr, \
                                      optiling::utils::OpRunInfo &rinfo) {                                          \
    const string optype = TbeGetOpType(para);                                                                       \
    OP_TILING_CHECK(op_info_ptr == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(optype, "op_info_ptr is nullptr."),     \
                    return false);                                                                                  \
    const std::shared_ptr<Tbe##oppattern##CustomCompileInfo> compile_ptr =                                          \
        std::dynamic_pointer_cast<Tbe##oppattern##CustomCompileInfo>(op_info_ptr);                                  \
    OP_TILING_CHECK(                                                                                                \
        compile_ptr == nullptr,                                                                                     \
        VECTOR_INNER_ERR_REPORT_TILIING(optype, "change CompileInfoBase to CustomCompileInfo failed."),             \
        return false);                                                                                              \
    return opfunc(#oppattern, para, compile_ptr->compile_info, rinfo);                                              \
  }                                                                                                                 \
  std::shared_ptr<CompileInfoBase> Tbe##oppattern##TilingV4CustomParsefunc(const ge::Operator &para,                \
                                                                           const ge::AscendString &compile_info) {  \
    const string optype = TbeGetOpType(para);                                                                       \
    std::shared_ptr<Tbe##oppattern##CustomCompileInfo> compile_ptr =                                                \
        ops::make_shared_nothrow<Tbe##oppattern##CustomCompileInfo>();                                              \
    if (compile_ptr == nullptr) {                                                                                   \
      OP_LOGE(optype, "make_shared failed, will return nullptr!");                                                  \
      return nullptr;                                                                                               \
    }                                                                                                               \
    try {                                                                                                           \
      bool parse_ret =                                                                                              \
          parse_func(optype, nlohmann::json::parse(compile_info.GetString()), compile_ptr->compile_info);           \
      if (parse_ret) {                                                                                              \
        return compile_ptr;                                                                                         \
      }                                                                                                             \
    } catch (nlohmann::json::parse_error& ex) {                                                                     \
      OP_LOGE(optype, "Failed to set compile_info_value from op_compile_info:%s", compile_info.GetString());        \
      return nullptr;                                                                                               \
    }                                                                                                               \
    OP_LOGE(optype, "do parse_func failed, will return nullptr!");                                                  \
    return nullptr;                                                                                                 \
  }

#define REGISTER_OP_TILING_V4_CUBE(oppattern, optype) \
  REGISTER_OP_TILING_V4(optype, Tbe##oppattern##TilingV4Custom, Tbe##oppattern##TilingV4CustomParsefunc)

namespace optiling {
using optiling::ByteBuffer;
using namespace ge;

const std::string PATTERN_REDUCE = "CommReduce";
const std::string PATTERN_ELEMWISE = "ElemWise";
const std::string PATTERN_BROADCAST = "Broadcast";
const std::string PATTERN_NORM = "Norm";
const std::string PATTERN_TRANSPOSE = "Transpose";

const std::map<std::string, std::int64_t> NO_OPTIONAL_VALUE;
const std::map<std::string, DataType> STR_TO_DATATYPE = {{"float", DT_FLOAT},
                                                         {"float32", DT_FLOAT},
                                                         {"float16", DT_FLOAT16},
                                                         {"int8", DT_INT8},
                                                         {"int16", DT_INT16},
                                                         {"int32", DT_INT32},
                                                         {"int64", DT_INT64},
                                                         {"uint8", DT_UINT8},
                                                         {"uint16", DT_UINT16},
                                                         {"uint32", DT_UINT32},
                                                         {"uint64", DT_UINT64},
                                                         {"bool", DT_BOOL},
                                                         {"double", DT_DOUBLE},
                                                         {"dual", DT_DUAL},
                                                         {"dual_sub_int8", DT_DUAL_SUB_INT8},
                                                         {"dual_sub_uint8", DT_DUAL_SUB_UINT8},
                                                         {"int4", DT_INT4},
                                                         {"bfloat16", DT_BF16}};

/*
 * @brief: read input shapes from paras
 * @param [in] paras: ge::Operator
 * @return vector<vector<int64_t>>: shapes vector of inputs
 */
vector<vector<int64_t>> GetInputShapes(const ge::Operator& paras);

/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
 * @return string: datatype string
 */

std::string to_string(const ge::DataType& type);
/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string to_string(const ge::Format& format);

/*
 * @brief: get string from TypedContinuousVector
 * @param [in] vec: TypedContinuousVector
 * @return string: string of TypedContinuousVector
 */
template<typename T>
std::string to_string(const gert::TypedContinuousVector<T>* vec) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < vec->GetSize(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << vec->GetData()[i];
  }
  oss << ")";
  return oss.str();
}

/*
 * @brief: get m and n greatest common divisor
 * @param [in] Tp: m, n
 * @return : Tp:
 */
template <typename Tp>
inline Tp Gcd(Tp m, Tp n) {
  while (n != 0)
  {
    Tp t = m % n;
    m = n;
    n = t;
  }
  return m;
}

/*
 * @brief: if shape is empty set {1}
 * @param [in] shape: std::vector<int64_t>
 * @return : void
 */
inline void ScalarToShape(std::vector<int64_t>& shape) {
  if (shape.empty())
    shape.push_back(1);
}

/*
 * @brief: get tensor shape size
 * @param [in] shape: tensor shape
 * @return : int64
 */
int64_t GetTensorSize(const GeShape& shape);

template <typename T>
string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result = "(";
  const T* data_addr = reinterpret_cast<const T*>(data.c_str());
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += ",";
  }
  result += ")";

  return result;
}

/*
 * @brief: get Byte size base on dtype(string)
 * @param [in] op_type: string dtype
 * @return int64_t: byte len
 */
int64_t GetByteLenByString(const std::string& op_type);

/*
 * @brief: get Byte size base on dtype(string)
 * @param [in] op_type: string dtype
 * @return DataType: ge dtype
 */
ge::DataType GetGeTypeFromStr(const std::string& dtype_str);

/*
 * @brief: get data block elements
 * @param [in] dtype: ge DataType
 * @return Int: dataBlock;
 */
int64_t GetDataBlockElems(const ge::DataType& dtype);

template <typename T>
bool GetCompileValue(const nlohmann::json& all_vars, const std::string& name, T& value) {
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
bool GetCompileValue(const nlohmann::json& all_vars, const std::string& name, T1& value, const T2 default_value) {
  if (!GetCompileValue(all_vars, name, value)) {
    value = static_cast<T1>(default_value);
  }
  return true;
}

#define OP_TILING_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                                  \
    try {                                               \
      exec_expr0;                                       \
    } catch (...) {                                     \
      exec_expr1;                                       \
    }                                                   \
  } while (0)

/*
 * @brief: transfor the json to vector_int64, with the json string key
 * @param [in] op: ge::Operator
 * @param [in] compile_info: ge::AscendString for compile info
 * @param [in] compile_info_key: the string vector, inclue the key value for op_type
 * @param [in] optional_key: the map for default compile info, set default value, when key isnot in compile_info
 * @return void*: ptr for vector result;
 */
void* ParseCompileToInt64Vec(const ge::Operator& op, const ge::AscendString compile_info,
                             const std::vector<std::string>& compile_info_key,
                             const std::map<std::string, int64_t>& optional_key);

/*
 * @brief: transfor the json to vector_int64, with the json string key
 * @param [in] op: ge::Operator
 * @param [in] compile_info: ge::AscendString for compile info
 * @param [in] compile_info_key: the string vector, inclue the key value for op_type
 * @param [in] optional_key: the map for default compile info, set default value, when key isnot in compile_info
 * @param [out] compile_vec: the compile parse result
 * @return bool: true or false;
 */
bool ParseCompileToInt64Vec(const ge::Operator& op, const ge::AscendString compile_info,
                            const std::vector<std::string>& compile_info_key,
                            const std::map<std::string, int64_t>& optional_key, std::vector<int64_t>& compile_vec);

bool AddReducMeanCof(const GeShape &input_shape, const DataType input_dtype,
                     const std::vector<int32_t>& reduce_axis, utils::OpRunInfo &run_info);
}  // namespace optiling
#endif  // CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_