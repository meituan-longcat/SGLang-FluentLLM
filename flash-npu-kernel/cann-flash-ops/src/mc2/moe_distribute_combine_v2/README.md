# MoeDistributeCombineV2

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

- 接口功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）。

    相较于MoeDistributeCombine算子，该算子变更如下：
    -   输入了更详细的token信息辅助`MoeDistributeCombineV2`高效地进行全卡同步，因此原算子中shape为(`Bs` * `K`,)的`expandIdx`入参替换为shape为(`A` * 128,)的`assistInfoForCombine`参数；
    -   新增`sharedExpertXOptional`入参，支持在`sharedExpertNum`为0时，由用户输入共享专家计算后的token；
    -   新增`commAlg`入参，代替`HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`环境变量。
    详细说明请参考以下参数说明。
- 计算公式：

    - 不存在TP域通信时：

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    - 存在TP域通信时：

    $$
    rsOut = ReduceScatterV(expandX)\\
    ataOut = AllToAllV(rsOut)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    注意该算子必须与MoeDistributeDispatchV2配套使用，相当于按MoeDistributeDispatchV2算子收集数据的路径原路返还。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>expandX</td>
   <td>输入</td>
   <td>根据expertIds进行扩展过的token特征。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assistInfoForCombine</td>
   <td>输入</td>
   <td>表示同一专家收到的token个数，对应MoeDistributeDispatchV2中的assistInfoForCombineOut输出。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>输入</td>
   <td>从EP通信域各卡接收的token数，对应MoeDistributeDispatchV2中的epRecvCounts输出。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>输入</td>
   <td>每个token的topK个专家的权重。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpSendCountsOptional</td>
   <td>可选输入</td>
   <td>从TP通信域各卡接收的token数，对应MoeDistributeDispatchV2中的tpRecvCounts输出，有TP域通信需传参，无TP域通信传空指针。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>可选输入</td>
   <td>表示token是否参与通信，可传有效数据或空指针；1D时true需排在false前（例：{true, false, true}非法），2D时token对应K个值全为false则不参与通信；默认所有token参与通信；各卡BS不一致时所有token需有效。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activationScaleOptional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScaleOptional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOptional</td>
   <td>可选输入</td>
   <td>表示本卡token的权重，对应MoeDistributeDispatchV2中的expandScales输出。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertXOptional</td>
   <td>可选输入</td>
   <td>表示共享专家计算后的token，数据类型需与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elasticInfoOptional</td>
   <td>可选输入</td>
   <td>EP通信域动态缩容信息。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>oriXOptional</td>
   <td>可选输入</td>
   <td>表示未经过FFN（Feed-Forward Neural network）的token数据，在使能copyExpert或使能constExpert的场景下需要本输入数据。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertAlpha1Optional</td>
   <td>可选输入</td>
   <td>在使能constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertAlpha2Optional</td>
   <td>可选输入</td>
   <td>在使能constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertVOptional</td>
   <td>可选输入</td>
   <td>在使能constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
  <tr>
   <td>groupEp</td>
   <td>属性</td>
   <td>EP通信域名称（专家并行通信域），字符串长度范围为[1, 128)，不能和groupTp相同。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>属性</td>
   <td>EP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>属性</td>
   <td>EP域本卡Id，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>属性</td>
   <td>MoE专家数量，满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>可选属性</td>
   <td><li>TP通信域名称（数据并行通信域）。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>可选属性</td>
   <td><li>TP通信域大小。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>可选属性</td>
   <td><li>TP域本卡Id，无TP域通信时传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>可选属性</td>
   <td><li>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</li><li>默认值为1。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡数量，取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>可选属性</td>
   <td><li>EP域全局的batch size大小；各rank Bs一致时，globalBs = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡Bs最大值）。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>outDtype</td>
   <td>可选属性</td>
   <td><li>用于指定输出x的数据类型，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>可选属性</td>
   <td><li>通信量化类型，取值范围0或2；0表示通信不量化，2表示通信int8量化。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListType</td>
   <td>可选属性</td>
   <td><li>groupList格式，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>可选属性</td>
   <td><li>表示通信亲和内存布局算法。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zeroExpertNum</td>
   <td>可选属性</td>
   <td><li>零专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copyExpertNum</td>
   <td>可选属性</td>
   <td><li>copy专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertNum</td>
   <td>可选属性</td>
   <td><li>常量专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xOut</td>
   <td>输出</td>
   <td>表示处理后的token，数据类型、数据格式与expandX保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

* <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    * 不支持共享专家场景，不支持`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`sharedExpertXOptional`。
    * 仅支持EP域，无TP域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性，且`tpRecvCounts`输出为无效内容。
    * 不支持动态缩容场景，不支持`elasticInfoOptional`。
    * 当`commAlg` = "hierarchy"，必须传入`expandScalesOptional`。
    * 不支持常量专家场景，不支持`constExpertNum`、`constExpertAlpha1Optional`、`constExpertAlpha2Optional`和`constExpertVOptional`，使用默认值即可。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    * 不支持`expandScalesOptional`。
    * 不支持`commAlg`。

## 约束说明

- `MoeDistributeDispatchV2`与`CombineV2`系列算子必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatchV2`的Tensor输出`assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut`中的元素值可能不同，使用时直接将上述Tensor传给`CombineV2`系列算子对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用算子过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commAlg`参数，`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致，且和`MoeDistributeDispatchV2`算子对应参数也保持一致。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足`A` = `Bs` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum`。
        - 对于MoE专家，当`globalBs`为0时，要满足`A` >= `Bs` * `epWorldSize` * min(`localExpertNum`, `K`)；当`globalBs`非0时，要满足`A` >= `globalBs` * min(`localExpertNum`, `K`)。
    - `K`：表示选取topK个专家，取值范围为0 < `K` ≤ 16同时满足0 < `K` ≤ `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`。
    - `localExpertNum`：表示本卡专家数量。
        - 对于共享专家卡，`localExpertNum` = 1
        - 对于MoE专家卡，`localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)，`localExpertNum` > 1时，不支持TP域通信。

- 参数约束：
    - `zeroExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[`moeExpertNum`, `moeExpertNum` + `zeroExpertNum`)。
    - `copyExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的copy专家的ID的值是[`moeExpertNum` + `zeroExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum`)。
    - `constExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的常量专家的ID的值是[`moeExpertNum` + `zeroExpertNum` + `copyExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`)。
    - `oriXOptional`：可选择传入有效数据或填空指针，当`copyExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为 (`Bs`, `H`)，数据类型需跟`expandX`保持一致。
    - `constExpertAlpha1Optional`：可选择传入有效数据或填空指针，当`constExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`constExpertNum`, )，数据类型需跟`expandX`保持一致。
    - `constExpertAlpha2Optional`：可选择传入有效数据或填空指针，当`constExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`constExpertNum`, )，数据类型需跟`expandX`保持一致。
    - `constExpertVOptional`：可选择传入有效数据或填空指针，当`constExpertNum`不为0或`constExpertNum`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`constExpertNum`, `H`)，数据类型需跟`expandX`保持一致。

- 本文公式中的"/"表示整除。
- 通信域使用约束：
    - 一个模型中的`MoeDistributeCombineV2`和`MoeDistributeDispatchV2`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`MoeDistributeCombineV2`和`MoeDistributeDispatchV2`仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - `commAlg`：当前版本支持nullptr， ""， "fullmesh"， "hierarchy"四种输入方式，若配置"hierarchy"，建议搭配25.0.RC1.1及以上版本驱动使用。
        - nullptr和""：仅在此场景下，`HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`配置生效。当`HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0时，调用"hierarchy"算法，否则调用"fullmesh"算法。不推荐使用该方式。
        - "fullmesh"：token数据直接通过RDMA方式发往topk个目标专家所在的卡。
        - "hierarchy"：token数据经过跨机、机内两次发送，仅不同server同号卡之间使用RDMA通信，server内使用HCCS通信。
    - `HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`：不推荐使用该环境变量控制通信算法，原`HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0场景，下文均通过`commAlg`="hierarchy"替代，默认场景使用`commAlg`="fullmesh"替代。
    - `commAlg` = "hierarchy"时，不支持`xActiveMaskOptional`、`oriXOptional`、`zeroExpertNum`、`copyExpertNum`。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围(0, 7168]，且保证是32的整数倍。
            - `commAlg` = "hierarchy"并且驱动版本≥25.0.RC1.1时支持(0, 10*1024]且为32的整数倍。
        - `Bs`：表示batch sequence size，即本卡最终输出的token数量。
            - `commAlg` = "fullmesh"：取值范围为[1, 256]。
            - `commAlg` = "hierarchy"：取值范围为[1, 512]。
    - 属性约束：
        - `epWorldSize`：依commAlg取值，"fullmesh"支持2、3、4、5、6、7、8、16、32、64、128、192、256、384；"hierarchy"支持16、32、64。
        - `moeExpertNum`：取值范围(0, 512]。
        - `commQuantMode`：2，开启通信int8量化，仅当`commAlg` = "hierarchy"且驱动版本不低于25.0.RC1.1时支持。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
        - `commAlg` = "fullmesh"：要求 >= (`Bs` * `epWorldSize` * min(`localExpertNum`, `K`) * `H` * 4B + 4MB)。
        - `commAlg` = "hierarchy"：要求 >= (`moeExpertNum` + `epWorldSize` / 4) * Align512(`maxBs` * (`H` * 2 + 16 * Align8(`K`))) * 1B + 8MB，其中Align8(x) = ((x + 8 - 1) / 8) * 8，Align512(x) = ((x + 512 - 1) / 512) * 512。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `Bs`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
    - 参数约束：
        - `epWorldSize`：取值支持8、16、32、64、128、144、256、288。
        - `moeExpertNum`：取值范围(0, 1024]。
        - `groupTp`：字符串长度范围为[1, 128)，不能和`groupEp`相同。
        - `tpWorldSize`：取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。
        - `tpRankId`：取值范围[0, 1]，同一个TP通信域中各卡的`tpRankId`不重复。无TP域通信时，传0即可。
        - `sharedExpertNum`：当前取值范围[0, 4]。
        - `commQuantMode`：int8量化当且仅当`tpWorldSize` < 2时可使能。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 2 * (`localExpertNum` * `maxBs` * `epWorldSize` * Align512(Align32(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBs` * Align512(2 * `H`))，`localExpertNum`需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) * 512，Align32(x) = ((x + 32 - 1) / 32) * 32。

- <term>Ascend 950PR/Ascend 950DT</term>：
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `Bs`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
    - 参数约束：
        - `epWorldSize`：取值支持[2, 768]。
        - `moeExpertNum`：取值范围(0, 1024]。
        - `groupTp`当前版本不支持，传空字符即可。
        - `tpWorldSize`当前版本不支持，传0即可。
        - `tpRankId`当前版本不支持，传0即可。
        - `expertShardType`当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - `sharedExpertNum`当前取值范围[0, 4]。
        - `sharedExpertRankNum`取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。
        - `commQuantMode`取值范围0或2（0表示不量化，2表示int8量化）。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 2 * (`localExpertNum` * `maxBs` * `epWorldSize` * Align512(Align32(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBs` * Align512(2 * `H`))，`localExpertNum`需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) * 512，Align32(x) = ((x + 32 - 1) / 32) * 32。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_combine_v2.cpp](./examples/test_aclnn_moe_distribute_combine_v2.cpp) | 通过[aclnnMoeDistributeCombineV2](./docs/aclnnMoeDistributeCombineV2.md)接口方式调用moe_distribute_combine_v2算子。 |
