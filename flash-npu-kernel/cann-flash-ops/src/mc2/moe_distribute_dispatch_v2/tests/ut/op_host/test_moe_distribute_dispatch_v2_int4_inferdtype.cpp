/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"

namespace MoeDistributeDispatchV2 {

TEST(MoeDistributeDispatchV2InferDtype, PackedInt4PreservesStorageDtype)
{
    ge::DataType xType = ge::DT_INT32;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType scalesType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 6)
        .NodeAttrs({
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        })
        .InputDataTypes({&xType, &expertIdsType, &scalesType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2")->infer_datatype;
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}

}  // namespace MoeDistributeDispatchV2
