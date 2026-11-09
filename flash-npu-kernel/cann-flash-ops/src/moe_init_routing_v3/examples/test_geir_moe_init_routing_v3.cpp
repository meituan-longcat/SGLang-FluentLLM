/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"
#include "nn_other.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

const int B = 3; // Batch size
const int S = 1; // Sequence length
const int H = 2; // Hidden size
const int K = 4;

#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                          \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0); \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);     \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenOnesData(placeholder##intputIndex##_shape,                                      \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        intputDtype,                                                                         \
        2);                                                                                  \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);           \
    input.push_back(tensor_placeholder##intputIndex);                                        \
    graph.AddOp(placeholder##intputIndex);                                                   \
    MIRV3op.set_input_##intputName(placeholder##intputIndex);                                \
    inputs.push_back(placeholder##intputIndex);

#define ADD_INPUT_ATTR(attrName, attrValue)                                                  \
    MIRV3op.set_attr_##attrName(attrValue);

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                    \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                  \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);     \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenOnesData(placeholder##intputIndex##_shape,                                      \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        intputDtype,                                                                         \
        2);                                                                                  \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);              \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);          \
    graph.AddOp(placeholder##intputIndex);                                                   \
    MIRV3op.set_input_##intputName(placeholder##intputIndex);                                \
    MIRV3op.update_input_desc_##intputName(placeholder##intputIndex##_desc);                 \
    inputs.push_back(placeholder##intputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                        \
    TensorDesc outputName##outputIndex##_desc =                                              \
        TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype);                          \
    MIRV3op.update_output_desc_##outputName(outputName##outputIndex##_desc);

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    float *pData = new (std::nothrow) float[size];

    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t *)pData, data_len);
    return SUCCESS;
}

int32_t GenOnesData(
    vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, DataType data_type, int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t *pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp;
    fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto MIRV3op = op::MoeInitRoutingV3("test_geir_moe_init_routing_v3");

    // shape定义
    std::vector<int64_t> x_shape = {B * S, H};
    std::vector<int64_t> expert_idx_shape = {B * S, K};
    std::vector<int64_t> scale_shape = {B * S};
    std::vector<int64_t> offset_shape = {1};

    std::vector<int64_t> expanded_x_shape = {B * S * K, H};
    std::vector<int64_t> expanded_row_idx_shape = {B * S * K};
    std::vector<int64_t> expert_tokens_count_or_cumsum_shape = {K};
    std::vector<int64_t> expanded_scale_shape = {B * S * K};

    std::vector<int64_t> active_expert_range_val = {0, 4};

    // 添加输入（顺序严格匹配 proto.h）
    ADD_INPUT(1, x, inDtype, x_shape);
    ADD_INPUT(2, expert_idx, DT_INT32, expert_idx_shape);
    ADD_INPUT(3, scale, DT_FLOAT, scale_shape);
    ADD_INPUT(4, offset, DT_FLOAT, offset_shape);

    // 添加输出（顺序严格匹配 proto.h）
    ADD_OUTPUT(1, expanded_x, DT_FLOAT16, expanded_x_shape);
    ADD_OUTPUT(2, expanded_row_idx, DT_INT32, expanded_row_idx_shape);
    ADD_OUTPUT(3, expert_tokens_count_or_cumsum, DT_INT64, expert_tokens_count_or_cumsum_shape);
    ADD_OUTPUT(4, expanded_scale, DT_FLOAT, expanded_scale_shape);

    ADD_INPUT_ATTR(active_num, 12);
    ADD_INPUT_ATTR(expert_capacity, 4);
    ADD_INPUT_ATTR(expert_num, 256);
    ADD_INPUT_ATTR(drop_pad_mode, 0);
    ADD_INPUT_ATTR(expert_tokens_num_type, 1);
    ADD_INPUT_ATTR(expert_tokens_num_flag, true);
    ADD_INPUT_ATTR(quant_mode, -1);
    ADD_INPUT_ATTR(active_expert_range, active_expert_range_val);
    ADD_INPUT_ATTR(row_idx_type, 1);

    outputs.push_back(MIRV3op);
    // 添加完毕
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    std::cout << argv[1] << std::endl;
    char *endptr;

    DataType inDtype = DT_FLOAT16;
    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)output_file.c_str(), data_size, output_data_i);
        int32_t *result = (int32_t*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, result[j]);
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}