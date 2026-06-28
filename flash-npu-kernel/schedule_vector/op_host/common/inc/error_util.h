/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file error_util.h
 * \brief
 */
#ifndef OPS_COMMON_INC_ERROR_UTIL_H_
#define OPS_COMMON_INC_ERROR_UTIL_H_

#include <sstream>
#include <string>
#include <vector>
#include "common/util/error_manager/error_manager.h"
#include "error_code.h"
#include "graph/operator.h"
#include "op_log.h"

#define AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg)                                      \
  do {                                                                                           \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", get_cstr(err_msg));                                    \
    REPORT_CALL_ERROR(GetViewErrorCodeStr(ViewErrorCode::AICPU_INFER_SHAPE_ERROR).c_str(), "%s", \
                      ConcatString("op[", op_name, "], ", err_msg).c_str());                     \
  } while (0)

#define AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)                                      \
  do {                                                                                            \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", get_cstr(err_msg));                                     \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::AICPU_INFER_SHAPE_ERROR).c_str(), "%s", \
                       ConcatString("op[", op_name, "], ", err_msg).c_str());                     \
  } while (0)

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)                                \
  do {                                                                                       \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", get_cstr(err_msg));                                \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::VECTOR_INNER_ERROR).c_str(), "%s", \
                       ConcatString("op[", op_name, "], ", err_msg).c_str());                \
  } while (0)

#define INFER_AXIS_TYPE_ERR_REPORT(op_name, err_msg, ...)                                            \
  do {                                                                                               \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                         \
    REPORT_INNER_ERROR("EZ9999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__); \
  } while (0)

#define VECTOR_FUSION_INNER_ERR_REPORT(op_name, err_msg, ...)                                              \
  do {                                                                                                     \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                               \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::VECTOR_INNER_ERROR).c_str(), "op[%s], " err_msg, \
                       get_cstr(get_op_info(op_name)), ##__VA_ARGS__);                                     \
  } while (0)

#define VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(op_name, ptr, ret, err_msg, ...) \
  do {                                                                           \
    if ((ptr) == nullptr) {                                                      \
      VECTOR_FUSION_INNER_ERR_REPORT(op_name, err_msg, ##__VA_ARGS__);           \
      return (ret);                                                              \
    }                                                                            \
  } while (0)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)                                                 \
  do {                                                                                               \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                         \
    REPORT_INNER_ERROR("E69999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__); \
  } while (0)

#define CUBE_CALL_ERR_REPORT(op_name, err_msg, ...)                                                 \
  do {                                                                                              \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                        \
    REPORT_CALL_ERROR("E69999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__); \
  } while (0)

#define CUBE_INNER_ERR_REPORT_PLUGIN(op_name, err_msg, ...)                                          \
  do {                                                                                               \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                         \
    REPORT_INNER_ERROR("E59999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__); \
  } while (0)

#define CUBE_CALL_ERR_REPORT_PLUGIN(op_name, err_msg, ...)                                          \
  do {                                                                                              \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                        \
    REPORT_CALL_ERROR("E59999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__); \
  } while (0)

namespace ge {

/*
 * get debug string of vector
 * param[in] v vector
 * return vector's debug string
 */
template <typename T>
std::string DebugString(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << "[";
  if (v.size() > 0) {
    for (size_t i = 0; i < v.size() - 1; ++i) {
      oss << v[i] << ", ";
    }
    oss << v[v.size() - 1];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string DebugString(const std::vector<std::pair<T, T>>& v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << "(" << v[i].first << ", " <<v[i].second << ")";
  }
  oss << "]";
  return oss.str();
}

inline std::ostream& operator << (std::ostream& os, const ge::Operator& op) {
  return os << get_op_info(op);
}

/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T>
std::string ConcatString(const T& arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(const T& arg, const Ts& ... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

template <typename T>
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}

std::string GetViewErrorCodeStr(ge::ViewErrorCode errCode);

std::string GetShapeErrMsg(uint32_t index, const std::string& wrong_shape, const std::string& correct_shape);
#define REPORT_GET_SHAPE_ERR(op_name, index, wrong_shape, correct_shape, ...)                              \
  do {                                                                                                     \
    OP_LOGE_WITHOUT_REPORT(op_name, GetShapeErrMsg(index, wrong_shape, correct_shape));                    \
    REPORT_INPUT_ERROR("EZ0001",                                                                           \
                       std::vector<std::string>({"op_name", "index", "incorrect_shape", "correct_shape"}), \
                       std::vector<std::string>({get_cstr(get_op_info(op_name)),                           \
                                                 std::to_string(index),                                    \
                                                 wrong_shape,                                              \
                                                 correct_shape}));                                         \
  } while (0)

std::string GetAttrValueErrMsg(const std::string& attr_name, const std::string& wrong_val,
                               const std::string& correct_val);
#define REPORT_GET_ATTR_VALUE_ERR(op_name, attr_name, wrong_val, correct_val, ...)                         \
  do {                                                                                                     \
    OP_LOGE_WITHOUT_REPORT(op_name, GetAttrValueErrMsg(attr_name, wrong_val, correct_val));                \
    REPORT_INPUT_ERROR("EZ0002",                                                                           \
                       std::vector<std::string>({"op_name", "attr_name", "incorrect_val", "correct_val"}), \
                       std::vector<std::string>({get_cstr(get_op_info(op_name)),                           \
                                                 attr_name,                                                \
                                                 wrong_val,                                                \
                                                 correct_val}));                                           \
  } while (0)

std::string GetAttrSizeErrMsg(const std::string& attr_name, const std::string& wrong_size,
                              const std::string& correct_size);
#define REPORT_GET_ATTR_SIZE_ERR(op_name, attr_name, wrong_size, correct_size, ...)                         \
  do {                                                                                                      \
    OP_LOGE_WITHOUT_REPORT(op_name, GetAttrValueErrMsg(attr_name, wrong_size, correct_size));               \
    REPORT_INPUT_ERROR("EZ0003",                                                                            \
                       std::vector<std::string>({"op_name", "attr_name", "incorrect_size", "correct_size"}),\
                       std::vector<std::string>({get_cstr(get_op_info(op_name)),                            \
                                                 attr_name,                                                 \
                                                 wrong_size,                                                \
                                                 correct_size}));                                           \
  } while (0)

std::string GetInputInvalidErrMsg(const std::string& param_name);
#define REPORT_GET_INPUT_ERR(op_name, param_name, ...)                                         \
  do {                                                                                         \
    OP_LOGE_WITHOUT_REPORT(op_name, GetInputInvalidErrMsg(param_name));                        \
    REPORT_INPUT_ERROR("EZ0004",                                                               \
                       std::vector<std::string>({"op_name", "param_name"}),                    \
                       std::vector<std::string>({get_cstr(get_op_info(op_name)), param_name}));\
  } while (0)

std::string GetShapeSizeErrMsg(uint32_t index, const std::string& wrong_shape_size,
                               const std::string& correct_shape_size);
#define REPORT_GET_SHAPE_SIZE_ERR(op_name, index, wrong_shape_size, correct_shape_size, ...)            \
  do {                                                                                                  \
    OP_LOGE_WITHOUT_REPORT(op_name, GetShapeSizeErrMsg(index, wrong_shape_size, correct_shape_size));   \
    REPORT_INPUT_ERROR("EZ0005",                                                                        \
                       std::vector<std::string>({"op_name", "index", "incorrect_size", "correct_size"}),\
                       std::vector<std::string>({get_cstr(get_op_info(op_name)),                        \
                                                 std::to_string(index),                                 \
                                                 wrong_shape_size,                                      \
                                                 correct_shape_size}));                                 \
  } while (0)

std::string GetInputFormatNotSupportErrMsg(const std::string& param_name, const std::string& expected_format_list,
                                           const std::string& data_format);
#define REPORT_GET_INPUT_FORMAT_ERR(op_name, param_name, data_format, expected_format_list, ...)                   \
  do {                                                                                                             \
    OP_LOGE_WITHOUT_REPORT(op_name, GetInputFormatNotSupportErrMsg(param_name, expected_format_list, data_format));\
    REPORT_INPUT_ERROR("EZ0006",                                                                                   \
                       std::vector<std::string>({"op_name", "param_name", "data_format", "expected_format_list"}), \
                       std::vector<std::string>({get_cstr(get_op_info(op_name)),                                   \
                                                 param_name,                                                       \
                                                 data_format,                                                      \
                                                 expected_format_list}));                                          \
  } while (0)

std::string GetInputDtypeNotSupportErrMsg(const std::string& param_name, const std::string& expected_dtype_list,
                                          const std::string& data_dtype);
#define REPORT_GET_INPUT_DTYPE_ERR(op_name, param_name, data_dtype, expected_dtype_list, ...)                   \
  do {                                                                                                          \
    OP_LOGE_WITHOUT_REPORT(op_name, GetInputDtypeNotSupportErrMsg(param_name, expected_dtype_list, data_dtype));\
    REPORT_INPUT_ERROR("EZ0007",                                                                                \
                       std::vector<std::string>({"op_name", "param_name", "data_dtype", "expected_dtype_list"}),\
                       std::vector<std::string>({get_cstr(get_op_info(op_name)),                                \
                                                 param_name,                                                    \
                                                 data_dtype,                                                    \
                                                 expected_dtype_list}));                                        \
  } while (0)

std::string GetInputDTypeErrMsg(const std::string& param_name, const std::string& expected_dtype,
                                const std::string& data_dtype);

std::string GetInputFormatErrMsg(const std::string& param_name, const std::string& expected_format,
                                 const std::string& data_format);

std::string SetAttrErrMsg(const std::string& param_name);
std::string UpdateParamErrMsg(const std::string& param_name);

template <typename T>
std::string GetParamOutRangeErrMsg(const std::string& param_name, const T& real_value, const T& min, const T& max);

std::string OtherErrMsg(const std::string& error_detail);

void TbeInputDataTypeErrReport(const std::string& op_name, const std::string& param_name,
                               const std::string& expected_dtype_list, const std::string& dtype);

void GeInfershapeErrReport(const std::string& op_name, const std::string& op_type, const std::string& value,
                           const std::string& reason);
/*
 * log common runtime error
 * param[in] opname op name
 * param[in] error description
 * return void
 */
void CommonRuntimeErrLog(const std::string& opname, const std::string& description);
}  // namespace ge

#endif  // OPS_COMMON_INC_ERROR_UTIL_H_
