# aclnnMoeDistributeDispatchV3

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：对token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。

    相较于`aclnnMoeDistributeDispatchV2`接口，该接口变更如下：
    - 新增支持特殊专家场景
        -   zeroExpertNum≠0：通过传入大于0的zeroExpertNum参数使能本特性。

            $$Moe(oriXOptional) = 0$$

        -   copyExpertNum≠0：通过传入大于0的copyExpertNum参数使能本特性，同时还需传入有效的oriXOptional参数。

            $$Moe(oriXOptional) = oriXOptional$$

        -   constExpertNum≠0：通过传入大于0的constExpertNum参数使能本特性，同时还需传入有效的oriXOptional、constExpertAlpha1Optional、constExpertAlpha2Optional、constExpertVOptional参数。

            $$Moe(oriXOptional) = constExpertAlpha1Optional * oriXOptional + constExpertAlpha2Optional * constExpertVOptional$$

        详细说明请参考以下参数说明。
        参数oriXOptional、constExpertAlpha1Optional、constExpertAlpha2Optional、constExpertVOptional见aclnnMoeDistributeCombineV3.md文档。

- 计算公式：

    - 情形1：如果不存在tp域通信。

    $$
    expandXOut = AllToAllV(X)\\
    $$

    - 情形2：如果存在tp域通信。

    $$
    allToAllOut = AllToAllV(X)\\
    expandXOut = AllGatherV(allToAllOut)\\
    $$

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该接口必须与`aclnnMoeDistributeCombineV3`配套使用。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该接口必须与`aclnnMoeDistributeCombineV3`或`aclnnMoeDistributeCombineAddRmsNormV2`配套使用。

> 说明：
> `aclnnMoeDistributeCombineV3`、`aclnnMoeDistributeCombineAddRmsNormV2`算子在后续文档中统称为**CombineV3系列算子**。



## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnMoeDistributeDispatchV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeDispatchV3”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeDispatchV3GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    const char*      groupTp,
    int64_t          tpWorldSize,
    int64_t          tpRankId,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          quantMode,
    int64_t          globalBs,
    int64_t          expertTokenNumsType,
    const char*      commAlg,
    int64_t          zeroExpertNum,
    int64_t          copyExpertNum,
    int64_t          constExpertNum,
    aclTensor*       expandXOut,
    aclTensor*       dynamicScalesOut,
    aclTensor*       assistInfoForCombineOut,
    aclTensor*       expertTokenNumsOut,
    aclTensor*       epRecvCountsOut,
    aclTensor*       tpRecvCountsOut,
    aclTensor*       expandScalesOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchV3(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchV3GetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px"> 
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>使用说明</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>本卡发送的token数据。</td>
    <td>2D Tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td><code>(Bs, H)</code>（Bs=batch size，H=hidden size）</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引。</td>
    <td>2D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(Bs, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>输入</td>
    <td>每个专家的量化平滑参数。</td>
    <td>2D Tensor，非量化场景传空指针；动态量化可传有效数据或空指针。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(sharedExpertNum + moeExpertNum, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>输入</td>
    <td>表示token是否参与通信。</td>
    <td>可选择传入有效数据或传入空指针。<br>当输入为1D时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入；<br>当输入为2D时，参数为true表示当前token对应的expert_ids参与通信。若当前token对应的K个BOOL值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。</td>
    <td>BOOL</td>
    <td>ND</td>
    <td><br>当输入1D时，shape为 <code>(BS,)</code>；当输入2D时，shape为 <code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>输入</td>
    <td>每个token的topK个专家权重。</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(Bs, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>输入</td>
    <td>EP通信域动态缩容信息。</td>
    <td>当某些通信卡因异常而从通信域中剔除，实际参与通信的卡数可从本参数中获取。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>EP通信域名称（专家并行）。</td>
    <td>字符串长度[1, 128)，不能和groupTp相同。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>输入</td>
    <td>EP域本卡ID。</td>
    <td>取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>输入</td>
    <td>MoE专家数量。</td>
    <td>满足 <code>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0</code>。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>输入</td>
    <td>TP通信域名称（数据并行）。</td>
    <td>-</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>输入</td>
    <td>TP通信域大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>输入</td>
    <td>TP域本卡ID。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>输入</td>
    <td>共享专家卡分布类型。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>输入</td>
    <td>共享专家数量（一个共享专家可以复制部署到多个卡上）。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>输入</td>
    <td>共享专家卡数量。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>输入</td>
    <td>量化模式。</td>
    <td>支持0：非量化，2：动态量化。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>输入</td>
    <td>EP域全局batch size。</li></td>
    <td><br> <li> 各卡Bs一致时：<code>globalBs = Bs * epWorldSize</code> 或 0；</li> <li> 各卡Bs不一致时：<code>globalBs = maxBs * epWorldSize</code>，其中maxBs为单卡Bs最大值。</li></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>输入</td>
    <td>输出<code>expertTokenNums</code>的语义类型。</td>
    <td>支持0：expertTokenNums中的输出为每个专家处理的token数的前缀和，1：expertTokenNums中的输出为每个专家处理的token数量。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>输入</td>
        <td>通信亲和内存布局算法。</td>
        <td>-</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>zeroExpertNum</td>
    <td>输入</td>
    <td>零专家数量。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>copyExpertNum</td>
    <td>输入</td>
    <td>拷贝专家数量。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>constExpertNum</td>
    <td>输入</td>
    <td>常量专家数量。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandXOut</td>
    <td>输出</td>
    <td>根据expertIds扩展过的token特征 。</td>
    <td>2D Tensor 。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>-</td>
    <td><code>(max(tpWorldSize, 1) * A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>输出</td>
    <td>动态量化场景的缩放参数。</td>
    <td>1D Tensor，仅当<code>quantMode=2</code>时输出。</td>
    <td>FLOAT32</td>
    <td>-</td>
    <td><code>(A,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>输出</td>
    <td>给同一专家发送的token个数（对应aclnnMoeDistributeCombineV3中的assistInfoForCombine）。</td>
    <td>1D Tensor。</td>
    <td>INT32</td>
    <td>-</td>
    <td><code>(A * 128,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertTokenNumsOut</td>
    <td>输出</td>
    <td>每个专家收到的token个数。</td>
    <td>1D Tensor。</td>
    <td>INT64</td>
    <td>-</td>
    <td><code>(localExpertNum,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epRecvCountsOut</td>
    <td>输出</td>
    <td>从EP通信域各卡接收的token数（对应aclnnMoeDistributeCombineV3中的epSendCounts）。</td>
    <td>1D Tensor。</td>
    <td>INT32</td>
    <td>-</td>
    <td><code>(moeExpertNum + 2 * globalBs * K * serverNum,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpRecvCountsOut</td>
    <td>输出</td>
    <td>从TP通信域各卡接收的token数（对应aclnnMoeDistributeCombineV3中的tpSendCounts）。</td>
    <td>-</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandScalesOut</td>
    <td>输出</td>
    <td>本卡输出token的权重（对应aclnnMoeDistributeCombineV3中的expertScalesOptional）。</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回Device侧需申请的workspace大小。</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回包含算子计算流程的op执行器。</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <details>
    <summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：</summary>

    - commAlg 支持nullptr、""、"fullmesh"、"hierarchy"；推荐配置"hierarchy"并搭配≥25.0.RC1.1版本驱动；nullptr和""依HCCL环境变量选择算法（不推荐）；"fullmesh"通过RDMA直传token；"hierarchy"经跨机、机内两次发送优化通信。
    - commAlg为"hierarchy"或HCCL_INTRA_PCIE_ENABLE=1且HCCL_INTRA_ROCE_ENABLE=0时，scalesOptional 需传nullptr。
    - xActiveMaskOptional 依commAlg取值，"fullmesh"要求为1D Tensor，shape为(Bs, )；true需排在false前（例：{true, false, true}非法）；"hierarchy"当前版本不支持，传空指针即可。
    - expertScalesOptional 要求为2D Tensor，shape为(Bs, K)。
    - epWorldSize 依commAlg取值，"fullmesh"支持2、3、4、5、6、7、8、16、32、64、128、192、256、384；"hierarchy"支持16、32、64。
    - moeExpertNum 取值范围(0, 512]。
    - groupTp 当前版本不支持，传空字符即可。
    - tpWorldSize、tpRankId、expertShardType、sharedExpertNum、sharedExpertRankNum 当前版本不支持，传0即可。
    - epRecvCountsOut 的shape为(moeExpertNum + 2 * globalBs * K * serverNum,)（前moeExpertNum个为接收token数，剩余为通信前reduce相关信息）。
    - 当前不支持TP域通信。
    - expandScalesOut 要求为1D Tensor，shape为(A,)。
    - quantMode 支持0（非量化）、2（动态量化）。
    - elasticInfoOptional 当前版本不支持，传空指针即可。
    - zeroExpertNum 当commAlg="fullmesh"时，取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>)。
    - copyExpertNum 当commAlg="fullmesh"时，取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的拷贝专家的ID的值是[<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>)。
    - constExpertNum 当前版本不支持，传0即可。
    </details>

    <details>
    <summary><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：</summary>

    - commAlg 支持""、"fullmesh_v1"、"fullmesh_v2"三种输入方式。""：默认值，使能fullmesh_v1模板；"fullmesh_v1"：使能fullmesh_v1模板；"fullmesh_v2"：使能fullmesh_v2模板，其中commAlg仅支持tpWorldSize为1场景。
    - xActiveMaskOptional 要求为1D或2D Tensor（1D时shape为(Bs, )，2D时shape为(Bs, K)）；1D时true需排在false前，2D时token对应K个值全为false则不参与通信。
    - expertScalesOptional 当前版本不支持，传空指针即可。
    - epWorldSize 取值范围[2, 768]。
    - moeExpertNum 取值范围(0, 1024]。
    - groupTp 字符串长度范围为[0, 128)，不能和groupEp相同，仅在无tp域通信时支持传空。
    - tpWorldSize 取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。
    - tpRankId 取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复；无TP域通信时传0即可。
    - expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum 当前取值范围[0, 4]。
    - sharedExpertRankNum 取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。
    - epRecvCountsOut 的shape为(epWorldSize * max(tpWorldSize, 1) * localExpertNum,)。
    - 有TP域通信时tpRecvCountsOut为1D shape Tensor，shape为(tpWorldSize,)。
    - expandScalesOut 当前版本不支持该输出。
    - quantMode 支持0（非量化）、2（动态量化）。
    - elasticInfoOptional 当前版本不支持，传空指针即可。
    - zeroExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是<code>[moeExpertNum, moeExpertNum + zeroExpertNum)</code>。
    - copyExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，专家ID范围<code>[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)</code>。
    - constExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，专家ID范围<code>[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)</code>。
    </details>

    <details>
    <summary><term>Ascend 950PR/Ascend 950DT</term>：</summary>

    - commAlg 支持""、"fullmesh_v1"、"fullmesh_v2"三种输入方式。""：默认值，使能fullmesh_v1模板；"fullmesh_v1"：使能fullmesh_v1模板；"fullmesh_v2"：使能fullmesh_v2模板，其中commAlg仅支持tpWorldSize为1场景。
    - xActiveMaskOptional 要求为1D或2D Tensor（1D时shape为(Bs, )，2D时shape为(Bs, K)）；1D时true需排在false前，2D时token对应K个值全为false则不参与通信。
    - expertScalesOptional 当前版本不支持，传空指针即可。
    - epWorldSize 取值范围[2, 768]。
    - moeExpertNum 取值范围(0, 1024]。
    - groupTp 当前版本不支持，传空字符即可。
    - tpWorldSize 当前版本不支持，传0即可。
    - tpRankId 当前版本不支持，传0即可。
    - expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum 当前取值范围[0, 4]。
    - sharedExpertRankNum 取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。
    - epRecvCountsOut 的shape为(epWorldSize * max(tpWorldSize, 1) * localExpertNum,)。
    - 有TP域通信时tpRecvCountsOut为1D shape Tensor，shape为(tpWorldSize,)。
    - expandScalesOut 当前版本不支持该输出。
    - quantMode 支持0（非量化）、1（静态量化）、2（pertoken动态量化）、3（pergroup动态量化）、4（mxfp8动态量化）。
    - elasticInfoOptional 当前版本不支持，传空指针即可。
    - zeroExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是<code>[moeExpertNum, moeExpertNum + zeroExpertNum)</code>。
    - copyExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，专家ID范围<code>[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)</code>。
    - constExpertNum 取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，专家ID范围<code>[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)</code>。
    </details>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。  

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>输入和输出的shape不在支持的范围内。</td>
    </tr>
    <tr>
        <td>参数的取值不在支持的范围。</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeDispatchV3

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeDispatchV3GetWorkspaceSize获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
    </tr>
    </tbody>
    </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeDistributeDispatchV3默认确定性实现。

- **接口配套约束**：
  - `aclnnMoeDistributeDispatchV3`与CombineV3系列算子必须配套使用，前者输出的`assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut`需直接传入后者对应参数，业务逻辑不可依赖这些Tensor的具体值。

- **参数一致性约束**：
  - 所有卡的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commAlg`参数及`HCCL_BUFFSIZE`取值需保持一致，且与CombineV3系列算子对应参数一致。

- **产品特定约束**：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明中的“本卡”均表示单DIE。

- **Shape变量约束**：
  | 变量         | 定义与取值范围                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | 表示本卡需要分发的最大token数量，取值范围如下：<ul> <li>对于共享专家，要满足A = Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum。</li> <li>对于MoE专家，当globalBs为0时，要满足A >= Bs * epWorldSize * min(localExpertNum, K)；当globalBs非0时，要满足A >= globalBs * min(localExpertNum, K)。|
  | H（hidden size） | 表示hidden size隐藏层大小。<ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：依commAlg取值，"fullmesh"支持(0, 7168]且为32的整数倍；"hierarchy"并且驱动版本≥25.0.RC1.1时支持(0, 10*1024]且为32的整数倍；</li> <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围[1024, 8192]。</li></ul> |
  | Bs           | 表示本卡最终输出token数。<ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：依commAlg取值，"fullmesh"取值范围为 (0 < Bs ≤ 256)；"hierarchy"并且驱动版本≥25.0.RC1.1时取值范围为 (0 < Bs ≤ 512)；</li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：0 < Bs ≤512，且当commAlg为"fullmesh_v2"时，需满足0 <Bs ≤256。 </li></ul> |
  | topK    | 表示选取topK个专家，取值范围为0 < K ≤16，且<code>0 < K ≤ moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum</code>。<br> <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当commAlg为"fullmesh_v2"时，取值范围为0 < K ≤ 12。|
  | serverNum    | 表示服务器节点数，仅支持2、4、8。<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品：仅该场景的shape使用了该变量。                                                  |
  | localExpertNum |  本卡专家数：<ul><li>对于共享专家卡，localExpertNum = 1；</li><li>对于MoE专家卡，localExpertNum = <code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>，localExpertNum > 1时不支持TP通信。 </li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：应满足 0 < localExpertNum * epWorldSize ≤ 2048。|

- **环境变量约束**：
  - **HCCL_BUFFSIZE**：

      调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
          - commAlg配置为""或nullptr：依照HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE环境变量配置，选择"fullmesh"或"hierarchy"公式。
          - commAlg配置为"fullmesh": 设置大小要求 >= 2 \* (Bs \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)。
          - commAlg配置为"hierarchy": 设置大小要求 >= (`moeExpertNum` + `epWorldSize` / 4) * Align512(`maxBs` * (`H` * 2 + 16 * Align8(`K`))) * 1B + 8MB，其中Align8(x) = ((x + 8 - 1) / 8) * 8，Align512(x) = ((x + 512 - 1) / 512) * 512。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品/Ascend 950PR/Ascend 950DT</term>：
          - ep通信域内，当commAlg为"fullmesh_v1"或空字符串或空指针时：设置大小要求取值满足 ≥ 2 * (localExpertNum * maxBs * epWorldSize * Align512(Align32(2 * H) + 64) + (K + sharedExpertNum) * maxBs * Align512(2 * H))。
          - ep通信域内，当commAlg为"fullmesh_v2"时：设置大小要求取值满足 ≥ 2 * (localExpertNum * maxBs * epWorldSize * 480Align512(Align32(2 * H) + 64) + (K + sharedExpertNum) * maxBs * Align512(2 * H))。
          - tp通信域内：设置大小要求 \>= (A \* Align512(Align32(h \* 2) + 44) + A \* Align512(h \* 2)) \* 2。
          - 其中`480Align512(x) = ((x + 480 - 1) / 480) * 512`，`Align512(x) = ((x + 512 - 1) / 512) * 512`，`Align32(x) = ((x + 32 - 1) / 32) * 32`。

  - **HCCL_INTRA_PCIE_ENABLE**和**HCCL_INTRA_ROCE_ENABLE**：
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该环境变量不再推荐使用，建议commAlg配置"hierarchy"。

- **通信域使用约束**：
  - 一个模型中的CombineV3系列算子和`aclnnMoeDistributeDispatchV3`仅支持相同EP通信域，且该通信域中不允许有其他算子。
  - 一个模型中的CombineV3系列算子和`aclnnMoeDistributeDispatchV3`仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。

- **组网约束**：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。

- **其他约束**：
  - 公式中的“/”表示整除。
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>。

## 调用示例

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：请参考[aclnnMoeDistributeDispatchV2](../docs/aclnnMoeDistributeDispatchV2.md)中调用示例的准备部分和示例代码，按照上文的约束说明重新设置涉及的变量，V3接口相较于V2接口新增的场景参数按上述参数说明传值即可。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
       
    具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- 示例代码如下，仅供参考
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v3.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v3.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while(0)

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 8;
    constexpr uint32_t TP_WORLD_SIZE = 1;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i +1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        char hcomTpName[128] = {0};
        ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchStream = %p, combineStream = %p, \
                    context = %p\n", args.rankId, hcomEpName, hcomTpName, args.dispatchStream, args.combineStream,                 \
                    args.context);

        int64_t Bs = 8;
        int64_t H = 7168;
        int64_t K = 3;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 1;
        int64_t sharedExpertRankNum = 1;
        int64_t moeExpertNum = 7;
        int64_t quantMode = 0;
        int64_t globalBs = Bs * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;
        int64_t zeroExpertNum = 1;
        int64_t copyExpertNum = 1;
        int64_t constExpertNum = 1;
        if (args.epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            A = globalBs / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBs * (localExpertNum < K ? localExpertNum : K);
        }

        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;

        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;
        void *residualXDeviceAddr = nullptr;
        void *sharedExpertXDeviceAddr = nullptr;

        void *elasticInfoDeviceAddr = nullptr;
        void *oriXDeviceAddr = nullptr;
        void *constExpertAlpha1DeviceAddr = nullptr;
        void *constExpertAlpha2DeviceAddr = nullptr;
        void *constExpertVDeviceAddr = nullptr;

        void *xOutDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;

        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;
        aclTensor *residualX = nullptr;
        aclTensor *sharedExpertX = nullptr;


        aclTensor *elasticInfo = nullptr;
        aclTensor *oriX = nullptr;
        aclTensor *constExpertAlpha1 = nullptr;
        aclTensor *constExpertAlpha2 = nullptr;
        aclTensor *constExpertV = nullptr;

        aclTensor *xOut = nullptr;

        //定义当前场景下各变量维度
        std::vector<int64_t> xShape{Bs, H};
        std::vector<int64_t> expertIdsShape{Bs, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{Bs, K};

        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> sharedExpertXShape{Bs, 1, H};


        std::vector<int64_t> elasticInfoShape{4 + EP_WORLD_SIZE * 2};
        std::vector<int64_t> oriXShape{Bs, H};
        std::vector<int64_t> constExpertAlpha1Shape{constExpertNum, H};
        std::vector<int64_t> constExpertAlpha2Shape{constExpertNum, H};
        std::vector<int64_t> constExpertVShape{constExpertNum, H};

        std::vector<int64_t> xOutShape{Bs, H};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);

        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);

        int64_t elasticInfoSize = GetShapeSize(elasticInfoShape);
        int64_t oriXSize = GetShapeSize(oriXShape);
        int64_t constExpertAlpha1Size = GetShapeSize(constExpertAlpha1Shape);
        int64_t constExpertAlpha2Size = GetShapeSize(constExpertAlpha2Shape);
        int64_t constExpertVSize = GetShapeSize(constExpertVShape);

        int64_t xOutShapeSize = GetShapeSize(xOutShape);

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);

        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);

        int32_t isElastic = 1;
        int32_t rankNumAfterElastic = 4;
        int32_t sharedExpertRankNumAfterElastic = sharedExpertRankNum;
        int32_t moeExpertNumAfterElastic = rankNumAfterElastic - sharedExpertRankNumAfterElastic;
        std::unordered_set<int16_t> availableRank{
            0, 1, /*2, 3, 4, 5,*/ 6, 7
        };
        std::vector<int32_t> elasticInfoHostData{
            isElastic, rankNumAfterElastic, sharedExpertRankNumAfterElastic, moeExpertNumAfterElastic,
            0, 1, -1, -1, -1, -1, 2, 3,
            0, 1, 6, 7, -1, -1, -1, -1
        };
        std::vector<int16_t> oriXHostData(oriXSize, 1);
        std::vector<int16_t> constExpertAlpha1HostData(constExpertAlpha1Size, 0);
        std::vector<int16_t> constExpertAlpha2HostData(constExpertAlpha2Size, 0);
        std::vector<int16_t> constExpertVHostData(constExpertVSize, 0);

        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);


        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(sharedExpertXHostData, sharedExpertXShape, &sharedExpertXDeviceAddr, aclDataType::ACL_BF16, &sharedExpertX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(elasticInfoHostData, elasticInfoShape, &elasticInfoDeviceAddr, aclDataType::ACL_INT32, &elasticInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(oriXHostData, oriXShape, &oriXDeviceAddr, aclDataType::ACL_BF16, &oriX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha1HostData, constExpertAlpha1Shape, &constExpertAlpha1DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha2HostData, constExpertAlpha2Shape, &constExpertAlpha2DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertVHostData, constExpertVShape, &constExpertVDeviceAddr, aclDataType::ACL_BF16, &constExpertV);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);



        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;
        /**************************************** 调用dispatch warm up********************************************/
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, nullptr, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** 调用dispatch ********************************************/
        if (availableRank.find(args.rankId) != availableRank.end()) {
            // 调用第一阶段接口
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                return ret);
        }
        /**************************************** 调用combine ********************************************/
        // 调用第一阶段接口
        if (availableRank.find(args.rankId) != availableRank.end()) {
        ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // 调用第二阶段接口
        ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
            return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineV3                      \
                    execute successfully.\n", args.rankId);
        }
        // 释放device资源
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }

        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (dynamicScales != nullptr) {
            aclDestroyTensor(dynamicScales);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (expertTokenNums != nullptr) {
            aclDestroyTensor(expertTokenNums);
        }
        if (epRecvCounts != nullptr) {
            aclDestroyTensor(epRecvCounts);
        }
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }
        if (expandScales != nullptr) {
            aclDestroyTensor(expandScales);
        }
        if (residualX != nullptr) {
            aclDestroyTensor(residualX);
        }
        if (sharedExpertX != nullptr) {
            aclDestroyTensor(sharedExpertX);
        }
        if (elasticInfo != nullptr) {
            aclDestroyTensor(elasticInfo);
        }
        if (oriX != nullptr) {
            aclDestroyTensor(oriX);
        }
        if (constExpertAlpha1 != nullptr) {
            aclDestroyTensor(constExpertAlpha1);
        }
        if (constExpertAlpha2 != nullptr) {
            aclDestroyTensor(constExpertAlpha2);
        }
        if (constExpertV != nullptr) {
            aclDestroyTensor(constExpertV);
        }

        if (xOut != nullptr) {
            aclDestroyTensor(xOut);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (expertScalesDeviceAddr != nullptr) {
            aclrtFree(expertScalesDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (dynamicScalesDeviceAddr != nullptr) {
            aclrtFree(dynamicScalesDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (expertTokenNumsDeviceAddr != nullptr) {
            aclrtFree(expertTokenNumsDeviceAddr);
        }
        if (epRecvCountsDeviceAddr != nullptr) {
            aclrtFree(epRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }
        if (sharedExpertXDeviceAddr != nullptr) {
            aclrtFree(sharedExpertXDeviceAddr);
        }

        if (elasticInfoDeviceAddr != nullptr) {
            aclrtFree(elasticInfoDeviceAddr);
        }
        if (oriXDeviceAddr != nullptr) {
            aclrtFree(oriXDeviceAddr);
        }
        if (constExpertAlpha1DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha1DeviceAddr);
        }
        if (constExpertAlpha2DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha2DeviceAddr);
        }
        if (constExpertVDeviceAddr != nullptr) {
            aclrtFree(constExpertVDeviceAddr);
        }

        if (xOutDeviceAddr != nullptr) {
            aclrtFree(xOutDeviceAddr);
        }

        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret); return ret);
        }

        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessDispatchAndCombine, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```