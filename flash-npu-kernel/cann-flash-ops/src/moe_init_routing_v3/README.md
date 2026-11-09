# MoeInitRoutingV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持不量化和动态量化模式。本接口针对V2接口[aclnnMoeInitRoutingV2](../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)做了如下功能变更，请根据实际情况选择合适的接口：

    1.增加动态量化功能，支持输出expendX的 int8动态量化输出。

    2.增加参数activeExpertRangeOptional，支持筛选有效范围内的expertId。

    3.删除属性expertTokensBeforeCapacityFlag、删除输出expertTokensBeforeCapacityOut（使用expertTokensCountOrCumsumOut进行输出）。

- 计算公式：

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

  3.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.计算quant结果：
    - 动态quant：
        - 若不输入scale：
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
        - 若输入scale:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

  5.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
    $$

  6.expandedRowIdxOut的有效元素数量availableIdxNum计算方式为，expertIdx中activeExpertRangeOptional范围内的元素的个数
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>MOE的输入，即token特征输入，对应公式中x。</td>
        <td>FLOAT32、FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdx</td>
        <td>输入</td>
        <td>每一行特征对应的K个处理专家，里面元素专家id不能超过专家数。对应公式中 expertIdx。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scaleOptional</td>
        <td>可选输入</td>
        <td>表示用于计算quant结果的参数。如果不输入表示计算时不使用scale，对应公式中scale。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>offsetOptional</td>
        <td>可选输入</td>
        <td>表示用于计算quant结果的偏移值。在非量化场景下和动态quant场景下不输入，对应公式中offsetOptional。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>activeNum</td>
        <td>属性</td>
        <td>表示总的最大处理row数，输出expandedXOut只有这么多行是有效的。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>属性</td>
        <td>表示每个专家能够处理的tokens数，取值范围大于等于0。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>属性</td>
        <td>表示专家数，expertTokensNumType为key\_value模式时，取值范围为[0, 5120], 其它模式取值范围[0, 10240]。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>属性</td>
        <td>表示是否为 DropPad 场景，取值为 0 和 1。
          <li>0：表示 Dropless 场景，该场景下不校验 expertCapacity。</li>
          <li>1：表示 DropPad 场景。</li></td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumType</td>
        <td>属性</td>
        <td>取值为0、1和2 。
          <li>0：表示 comsum 模式。</li>
          <li>1：表示 count 模式，即输出的值为各个专家处理的 token 数量的累计值。</li>
          <li>2：表示 key\_value 模式，即输出的值为专家和对应专家处理 token 数量的累计值。</li></td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumFlag</td>
        <td>属性</td>
        <td>取值为false和true。
          <li>false：表示不输出 expertTokensCountOrCumsumOut。</li>
          <li>true：表示输出 expertTokensCountOrCumsumOut。</li></td>
        <td>Bool</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>属性</td>
        <td>取值为0、1、-1、2、3。
          <li>0：表示静态quant场景。</li>
          <li>1：表示动态quant场景。</li>
          <li>-1：表示不量化场景。</li>
          <li>2：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E5M2。</li>
          <li>3：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E4M3FN。</li>
        </td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeExpertRangeOptional</td>
        <td>可选属性</td>
        <td>长度为2，数组内的值为[expertStart, expertEnd], 表示活跃的expert范围在expertStart和expertEnd之间，左闭右开。要求值大于等于0，并且expertEnd不大于expertNum。</td>
        <td>ListInt</td>
        <td>-</td>
      </tr>
      <tr>
        <td>rowIdxType</td>
        <td>属性</td>
        <td>表示expandedRowIdxOut使用的索引类型，取值为0、1。（性能模板仅支持1）
          <li>0：表示gather类型的索引。</li>
          <li>1：表示scatter类型的索引。</li></td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedXOut</td>
        <td>输出</td>
        <td>根据expertIdx进行扩展过的特征。非量化场景下数据类型同x，量化场景quantMode为0、1时数据类型支持INT8，quantMode为2、3时数据类型分别支持FLOAT8_E5M2、FLOAT8_E4M3FN。</td>
        <td>FLOAT32、FLOAT16、BFLOAT16、INT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdxOut</td>
        <td>输出</td>
        <td>expandedXOut和x的索引映射关系，前availableIdxNum\*H个元素为有效数据，其余无效数据，当rowIdxType为0时，无效数据由-1填充；当rowIdxType为1时，无效数据未初始化。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertTokensCountOrCumsumOut</td>
        <td>输出</td>
        <td>
          <li>在expertTokensNumType为1的场景下，表示activeExpertRangeOptional范围内expert对应的处理token的总数。</li>
          <li>在expertTokensNumType为2的场景下，表示activeExpertRangeOptional范围内token总数为非0的expert，以及对应expert处理token的总数。</li></td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedScaleOut</td>
        <td>输出</td>
        <td>输出量化计算过程中scaleOptional的中间值。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 输入值域限制：
  - activeNum 当前未使用，校验需等于NUM_ROWS*K。
  - expertCapacity 当前未使用，仅校验非空。
  - dropPadMode 当前只支持0，代表 Dropless 场景。
  - expertTokensNumType 当前只支持 1 和 2，分别代表 count 模式和 key\_value 模式。
  - expertTokensNumFlag 只支持 true，代表输出 expertTokensCountOrCumsumOut。
  - quantMode:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持1、-1，分别代表动态量化场景和不量化场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持-1、1、2、3，分别表示不量化、动态量化、MXFP8量化到FLOAT8_E5m2、MXFP8量化到FLOAT8_E4M3FN。

- 其他限制：该算子部分产品支持两种性能模板，进入两种性能模板需要分别额外满足以下条件，不满足条件则进入通用模板。
  - 支持性能模板的产品：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品。</term>
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品。</term>

  - 进入低时延性能模板需要同时满足以下条件：
    - x、expertIdx、scaleOptional 输入 Shape 要求分别为：(1, 7168)、(1, 8)、(256, 7168)。
    - x 数据类型要求：BFLOAT16。
    - 属性要求：activeExpertRangeOptional=[0, 256]、 quantMode=1、expertTokensNumType=2、expertNum=256。

  - 进入大 batch 性能模板需要同时满足以下条件：
    - NUM_ROWS范围为[384, 8192]
    - K=8
    - expertNum=256
    - expertEnd-expertStart<=32
    - quantMode=-1
    - rowIdxType=1
    - expertTokensNumType=1
