# aclnnGroupedMatmulV3

**须知：该接口后续版本会废弃，请使用最新aclnnGroupedMatmulV5接口。**

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。输入输出数据类型均为aclTensorList，对应的功能为：

  - k轴分组：$k_i$各不相同，但$m_i/n_i$每组相同，此时$x_i/weight_i$可以在$k_i$上拼接。
  - m轴分组：$k_i$各组相同，$weight_i/y_i$可以在$n_i$上拼接。

  相较于[GroupedMatmul](aclnnGroupedMatmul.md)接口，**此接口新增：**
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
      - 非量化场景，支持weight转置（转置指若shape为[M,K]时，则stride为[1,M],数据排布为[K,M]的场景）。
      - 支持m轴和k轴分组，由groupType表示。
      - x、weight、y都为单tensor，非量化场景下，支持x、weight输入都为float32类型。
      - 量化、伪量化场景，支持weight转置，支持weight为单tensor。
      - 对于[aclnnGroupedMatmulGetWorkspaceSize](aclnnGroupedMatmul.md)接口支持的特性，该接口不支持x为单tensor，weight/y为多tensor场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：
      - 伪量化场景，支持weight转置，支持x，weight，y均为单tensor。
      - 对于[aclnnGroupedMatmulGetWorkspaceSize](aclnnGroupedMatmul.md)接口支持的特性，该接口不支持x为单tensor，weight/y为多tensor场景。

**说明：**

  - 单tensor指一个tensor list中所有分组的tensor在groupType指定的分组轴上合并为1个；否则为多tensor。
  - tensor转置：指若tensor shape为[M,K]时，则stride为[1,M],数据排布为[K,M]的场景，即非连续tensor。

- 计算公式：
  - **非量化场景：**

    $$
     y_i=x_i\times weight_i + bias_i
    $$

  - **量化场景：**

    $$
     y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
    $$

  - **反量化场景：**

    $$
     y_i=(x_i\times weight_i + bias_i) * scale_i
    $$

  - **伪量化场景：**

    $$
     y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedMatmulV3GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmulV3”接口执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulV3GetWorkspaceSize(
  const aclTensorList   *x,
  const aclTensorList   *weight,
  const aclTensorList   *biasOptional,
  const aclTensorList   *scaleOptional,
  const aclTensorList   *offsetOptional,
  const aclTensorList   *antiquantScaleOptional,
  const aclTensorList   *antiquantOffsetOptional,
  const aclTensor       *groupListOptional,
  int64_t                splitItem,
  int64_t                groupType,
  const aclTensorList   *y,
  uint64_t              *workspaceSize,
  aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedMatmulV3GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1483px"><colgroup>
  <col style="width: 210px">
  <col style="width: 90px">
  <col style="width: 370px">
  <col style="width: 232px">
  <col style="width: 339px">
  <col style="width: 86px">
  <col style="width: 92px">
  <col style="width: 64px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
      <td>tensorList长度支持[1, 128]或者[1, 1024]。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>公式中的weight。</td>
      <td>tensorList长度支持[1, 128]或者[1, 1024]。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>公式中的bias。</td>
      <td>长度与weight相同。</td>
      <td>INT32、BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>代表量化参数中的缩放因子。</td>
      <td>一般情况下，长度与weight相同。</td>
      <td>UINT64、INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>代表量化参数中的偏移量。</td>
      <td>长度与weight相同。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>输入</td>
      <td>代表伪量化参数中的缩放因子。</td>
      <td>长度与weight相同。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>输入</td>
      <td>代表伪量化参数中的偏移量。</td>
      <td>长度与weight相同。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional</td>
      <td>输入</td>
      <td>代表输入和输出分组轴方向的matmul大小分布。</td>
      <td>当输出中TensorList的长度为1时，groupListOptional中的最后一个值约束了输出数据的有效部分，groupListOptional中未指定的部分将不会参与更新。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>splitItem</td>
      <td>输入</td>
      <td>整数型参数，代表输出是否要做tensor切分。</td>
      <td>
      0/1代表输出为多tensor
      2/3代表输出为单tensor。
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType</td>
      <td>输入</td>
      <td>整数型参数，代表需要分组的轴。</td>
      <td>如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，2：k轴分组。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。</td>
      <td>tensorList长度支持[1, 128]或者[1, 1024]。</td>
      <td>FLOAT16、BFLOAT16、INT8、FLOAT32、INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - x支持FLOAT16、BFLOAT16、INT8、FLOAT32
    - weight支持FLOAT16、BFLOAT16、INT8、FLOAT32
    - biasOptional支持FLOAT16、FLOAT32、INT32
    - y支持FLOAT16、BFLOAT16、INT8、FLOAT32
    - 输入参数x、weight，输出参数y支持最多128个tensor。
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - x支持FLOAT16、BFLOAT16、FLOAT32、INT8
    - weight支持FLOAT16、BFLOAT16、FLOAT32、INT8
    - biasOptional支持FLOAT16、BFLOAT16、FLOAT32、INT32
    - y支持FLOAT16、BFLOAT16、FLOAT32、INT8
    - 不支持offsetOptional
    - groupType支持m轴分组和不分组，仅非量化支持k轴分组。
    - 输入参数x、weight，输出参数y在非量化场景支持最多1024个tensor，在伪量化场景支持最多128个tensor，在量化场景支持最多1个tensor。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1055px"><colgroup>
  <col style="width: 242px">
  <col style="width: 78px">
  <col style="width: 735px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、out的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>weight的长度不在支持范围。</td>
    </tr>
    <tr>
      <td>若bias不为空，bias的长度不等于weight的长度。</td>
    </tr>
    <tr>
      <td>groupListOptional维度不在支持范围内。</td>
    </tr>
    <tr>
      <td>splitItem为2、3的场景，out长度不等于1。</td>
    </tr>
    <tr>
      <td>splitItem为0、1的场景，out长度不等于weight的长度，groupListOptional长度不等于weight的长度。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulV3

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 834px"><colgroup>
    <col style="width: 118px">
    <col style="width: 87px">
    <col style="width: 629px">
    </colgroup>
    <thead>
      <tr>
        <th>参数说明</th>
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulV3GetWorkspaceSize获取。</td>
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

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnGroupedMatmulV3默认确定性实现。
- 如果传入groupListOptional，groupListOptional必须为非负递增数列。
- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 非量化场景支持的输入类型为：
    - x为FLOAT16、weight为FLOAT16、biasOptional为FLOAT16、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为FLOAT16；
    - x为BFLOAT16、weight为BFLOAT16、biasOptional为FLOAT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为BFLOAT16；
    - x为FLOAT32、weight为FLOAT32、biasOptional为FLOAT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为FLOAT32（仅x、weight、y都为单tensor场景支持）；
  - 量化场景支持的输入类型为：
    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为UINT64、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为INT8；
  - 伪量化场景支持的输入类型为：
    - x为FLOAT16、weight为INT8、biasOptional为FLOAT16、scaleOptional为空，offsetOptional为空，antiquantScaleOptional为FLOAT16、antiquantOffsetOptional为FLOAT16、y为FLOAT16；
    - x为BFLOAT16、weight为INT8、biasOptional为FLOAT32、scaleOptional为空，offsetOptional为空，antiquantScaleOptional为BFLOAT16、antiquantOffsetOptional为BFLOAT16、y为BFLOAT16；
  - 不同groupType支持场景：
    - a16w8、a16w4场景仅支持groupType为-1和0场景。
    - A8W8、A8W4、A4W4场景仅支持groupType为0场景中x tensor数为单。
    - x、weight、y的输入类型为aclTensorList，表示一个aclTensor类型的数组对象。下面表格支持场景用"单"表示由一个aclTensor组成的aclTensorList，"多"表示由多个aclTensor组成的aclTensorList。例如"单多单"，分别表示x为单tensor、weight为多tensor、y为单tensor。

      | groupType | x tensor数 | weight tensor数 | y tensor数 | splitItem| groupListOptional | 转置 | 其余场景限制 |
      |:---------:|:-------:|:-------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | 多个|多个|多个 | 0/1 | groupListOptional必须传空 | 1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一| x中tensor要求维度一致，支持2维，weight中tensor需为2维，y中tensor维度和x保持一致 |
      | 0 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）groupListOptional第1维最大支持1024，即最多支持1024个group |1）x不支持转置；<br> 2）支持weight转置，A8W4与A4W4场景不支持weight转置 |weight中tensor需为3维，x，y中tensor需为2维|
      | 0 | 单个|多个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）groupListOptional第1维最大支持128，即最多支持128个group|1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一 |1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 0 | 多个|多个|单个 | 2/3 | 1）groupListOptional可选；<br> 2）groupListOptional第1维最大支持128，即最多支持128个group |1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一|1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 2 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）groupListOptional第1维最大支持1024， 即最多支持1024个group | 1）x必须转置；<br> 2）weight不能转置 |1）x，weight中tensor需为2维，y中tensor需为3维；<br> 2）bias必须传空|
      | 2 | 单个|多个|多个 | 0/1 | groupListOptional必须传空 | 1）x必须转置；<br> 2）weight不能转置| 1）x，weight，y中tensor需为2维。<br> 2）weight长度最大支持128，即最多支持128个group；<br> 3）原始shape中weight每个tensor的第一维之和不应超过x第一维；<br> 4）bias必须传空 |

  - x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当x不转置时$x_i$的K轴或当x转置时$x_i$的M轴。$weight_i$的最后一维指当weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。

- <term>Ascend 950PR/Ascend 950DT</term>：

  <details>
    <summary><term>非量化量化场景约束</term></summary>
      <a id="非量化场景约束"></a>

  - 非量化场景支持的数据类型为：
    - 以下入参为空：scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional
    - 不为空的参数支持的数据类型组合要满足下表：

        |groupType| x       | weight  | biasOptional | y     |
        |:-------:|:-------:|:-------:| :------      |:------ |
        |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
        |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
        |-1/0/2   |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|

  </details>

  <details>
  <summary><term>伪量化量化场景约束</term></summary>
  <a id="伪量化场景约束"></a>

  - 伪量化场景支持的数据类型为：
    - 以下入参为空：scaleOptional、offsetOptional
    - 不为空的参数支持的数据类型组合要满足下表：

      |groupType| x       | weight  | antiquantScaleOptional |   antiquantOffsetOptional | biasOptional | y     |
      |:-------:|:-------:|:-------:| :------  | :------ | :------     |:------ |
      |-1/0   |BFLOAT16     |INT8   |BFLOAT16  |BFLOAT16/null  |BFLOAT16/FLOAT32/null   | BFLOAT16|
      |-1/0   |FLOAT16     |INT8  |FLOAT16 |FLOAT16/null   |FLOAT16/null| FLOAT16|

    - antiquantScaleOptional和非空的biasOptional、antiquantOffsetOptional要满足下表（其中g为matmul组数即分组数）：

      |groupType| 使用场景 | shape限制 |
      |:---------:|:---------:| :------ |
      |-1|weight多tensor|每个tensor 1维，shape为（$n_i$），不允许存在一个tensorList中部分tensor的shape为（$n_i$）部分tensor为空的情况 |
      |0|weight单tensor|每个tensor 2维，shape为（g, N）|

    - 仅支持单单单和多多多场景
  </details>

  <details>
  <summary>量化场景约束</summary>

  - 以下入参为空：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional
  - 仅支持单单单场景

  - 不为空的参数支持的数据类型和维度组合要满足下表（其中g为matmul组数即分组数）：

      |groupType| x dtype     | x shape | weight dtype |weight shape| biasOptional dtype |biasOptional shape| scaleOptional dtpye |scaleOptional shape| out dtpye    |out shape|
      |:-------:|:-------:|:-------:|:-------:|:-------:| :------      | :------      |:-------       | :------ |:-------       | :------ |
      |0|INT8     |(M,K)|INT8     |(g,K,N)/(g,N,K)|INT32/null    | (g,N)|UINT64/INT64  |(g,N)/null|INT8|(M,N)|

  </details>

  <details>
  <summary><term>groupType场景约束</term></summary>
  <a id="groupType场景约束"></a>

  - 不同groupType支持场景：
    - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，y，例如单多单表示支持x为单tensor，weight多tensor，y单tensor的场景。

        | groupType | 支持场景 | 场景限制 |
        |:---------:|:-------:| :-------|
        | -1 | 多多多 |1）仅支持splitItem为0/1<br>2）非量化x，out中tensor需为2维，shape分别为（$m_i$, $k_i$）和（$m_i$, $n_i$）；伪量化场景x中tensor要求维度一致，支持2-6维，y中tensor维度和x保持一致；weight中tensor需为2维，shape为（$n_i$, $k_i$）或（$k_i$, $n_i$）；bias中tensor需为1维，shape为（$n_i$）<br>3）groupListOptional必须传空<br>4）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>5）x不支持转置<br>6）仅支持ND进ND出 |
        | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，shape为（g, N, K）或（g, K, N）；x，y中tensor需为2维，shape分别为（M, K）和（M, N）；bias中tensor需为2维，shape为（g, N）<br>3）必须传groupListOptional，最后一个值不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持weight转置<br>6）x不支持转置<br>7）仅支持ND进ND出 |
        | 0 | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，最后一个值与x中tensor的第一维相等，且长度最大为1024<br>3）x，y中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>4）weight中每个tensor的N轴必须相等<br>5）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置<br>7）仅支持ND进ND出<br>8）仅支持非量化|
        | 0 | 多多单 |1）仅支持splitItem为2<br>2）x，y中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，groupListOptional的差值需与x中tensor的第一维一一对应，且长度最大为1024<br>5）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置<br>7）仅支持ND进ND出<br>8）仅支持非量化|
        | 2 | 单单单 |1）仅支持splitItem为2/3<br>2）x，weight中tensor需为2维，shape分别为（K, M）和（K, N）；y中tensor需为3维, shape为（g, M, N）<br>3）必须传groupListOptional，最后一个值不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）x必须转置且weight不能转置<br>6）仅支持ND进ND出<br>7）仅支持非量化|

      </details>

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v3.h"

#define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    std::vector<T> hostData(size, 0);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    std::vector<aclTensor*> tensors(size);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors.data() + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}


int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<std::vector<int64_t>> xShape = {{512, 256}};
    std::vector<std::vector<int64_t>> weightShape= {{2, 256, 256}};
    std::vector<std::vector<int64_t>> biasShape = {{2, 256}};
    std::vector<std::vector<int64_t>> yShape = {{512, 256}};
    std::vector<int64_t> groupListShape = {{2}};
    std::vector<int64_t> groupListData = {256, 512};
    void* xDeviceAddr[1];
    void* weightDeviceAddr[1];
    void* biasDeviceAddr[1];
    void* yDeviceAddr[1];
    void* groupListDeviceAddr;
    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclTensor* groupedList = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* y = nullptr;
    int64_t splitItem = 2;
    int64_t groupType = 0;

    // 创建x aclTensorList
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensorList
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensorList
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensorList
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建group_list aclTensor
    ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulV3第一段接口
    ret = aclnnGroupedMatmulV3GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, groupType, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulV3第二段接口
    ret = aclnnGroupedMatmulV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    for (int i = 0; i < 1; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
        }
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    for (int i = 0; i < 1; i++) {
        aclrtFree(xDeviceAddr[i]);
        aclrtFree(weightDeviceAddr[i]);
        aclrtFree(biasDeviceAddr[i]);
        aclrtFree(yDeviceAddr[i]);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
  ```
