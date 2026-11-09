# GroupedMatmul

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

-   算子功能：实现分组矩阵乘计算。如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数。当前支持m轴和k轴分组，对应的功能为：

    - m轴分组：$k_i$、$n_i$各组相同，$m_i$可以不相同。
    - k轴分组：$m_i$、$n_i$各组相同，$k_i$可以不相同。

-   计算公式：
    - **非量化场景：**
    $$
     y_i=x_i\times weight_i + bias_i
    $$

    - **量化场景（无perTokenScaleOptional）：**
    $$
      y_i=(x_i\times weight_i) * scale_i + offset_i
    $$
      - x为INT8，bias为INT32
      $$
        y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
      $$

    - **量化场景（有perTokenScaleOptional）：**
    $$
     y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i
    $$
      - x为INT8，bias为INT32
      $$
        y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
      $$

    - **反量化场景：**
    $$
     y_i=(x_i\times weight_i + bias_i) * scale_i
    $$

    - **伪量化场景：**
    $$
     y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$
      - x为INT8，weight为INT4（仅支持x、weight、y均为单tensor的场景）。其中$bias$为必选参数，是离线计算的辅助结果，且 $bias_i=8\times weight_i  * scale_i$ ，并沿k轴规约。
    $$
      y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i
    $$


## 参数说明
<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的输入`x`。</td>
      <td style="white-space: nowrap">FLOAT<sup>1</sup>、FLOAT16、INT16<sup>1</sup>、INT8、INT4<sup>1</sup>、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup></td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weight</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的`weight`。</td>
      <td style="white-space: nowrap">FLOAT<sup>1</sup>、FLOAT16、INT16<sup>1</sup>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup></td>
      <td style="white-space: nowrap">ND/NZ</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">biasOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的`bias`。</td>
      <td style="white-space: nowrap">FLOAT、FLOAT16、INT32、BFLOAT16<sup>2</sup></td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的`scale`，代表量化参数中的缩放因子。</td>
      <td style="white-space: nowrap">FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<sup>2</sup>、INT64<sup>2</sup></td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">offsetOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的`offset`，代表量化参数中的偏移量。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">antiquantScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的`antiquant_scale`，代表伪量化参数中的缩放因子。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">antiquantOffsetOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的`antiquant_offset`，代表伪量化参数中的缩放因子。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">perTokenScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的`per_token_scale`，代表量化参数中的由x量化引入的缩放因子。</td>
      <td style="white-space: nowrap">FLOAT、FLOAT8_E8M0<sup>2</sup></td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">代表输入和输出分组轴方向的matmul大小分布。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationInputOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">代表激活函数的反向输入，当前只支持传入nullptr。</td>
      <td style="white-space: nowrap">-</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationQuantScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">当前只支持传入nullptr。</td>
      <td style="white-space: nowrap">-</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationQuantOffsetOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">当前只支持传入nullptr。</td>
      <td style="white-space: nowrap">-</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">splitItem</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">代表输出是否要做tensor切分。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">代表需要分组的轴。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">代表groupList输入的分组方式。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">actType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">代表激活函数类型。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">tuningConfigOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">代表各个专家处理的token数的预期值，用于优化tiling。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">out</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">公式中的输出`y`。</td>
      <td style="white-space: nowrap">FLOAT、FLOAT16、INT32<sup>1</sup>、INT8<sup>1</sup>、BFLOAT16</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationFeatureOutOptional</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">激活函数的输入数据，当前只支持传入nullptr。</td>
      <td style="white-space: nowrap">-</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">dynQuantScaleOutOptional</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">当前只支持传入nullptr。</td>
      <td style="white-space: nowrap">-</td>
      <td style="white-space: nowrap">-</td>
    </tr>
  </tbody>
</table>


- <term>Ascend 950PR/Ascend 950DT AI处理器</term>：

  - 上表数据类型列中的角标“1”代表该系列不支持的数据类型。
  - 输入参数x、weight均不支持INT16类型，且x不支持int4类型。
  - 输出参数out不支持INT32、INT8类型。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  - 上表数据类型列中的角标“2”代表该系列不支持的数据类型。
  - 不支持FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0类型。
  - 输入参数biasOptional不支持BFLOAT16。
  - 输入参数scaleOptional不支持INT64类型。

## 约束说明
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 输入数据类型和格式如下说明（1.除weight外，其余格式都为ND；2.groupList是否传值与使用场景有关，具体请参考<a href="#groupType-constraints">groupType支持场景</a>约束）：

      |  类型  |    x    |    weight       |  bias        | scale  | offset     | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out     |
      |--------|---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | 非量化 | FLOAT   | FLOAT (ND)      | FLOAT/null   | null   | null       | null           | null            | null          | INT64     | null            | null                 | null                  | FLOAT   |
      | 非量化 | FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | null   | null       | null           | null            | null          | INT64     | null            | null                 | null                  | FLOAT16 |
      | 非量化 | BFLOAT16| BFLOAT16(ND/NZ) | FLOAT/null   | null   | null       | null           | null            | null          | INT64     | null            | null                 | null                  | BFLOAT16|
      | 伪量化 | FLOAT16 | INT8 (ND)       | FLOAT16/null | null   | null       | FLOAT16        | FLOAT16         | null          | INT64     | null            | null                 | null                  | FLOAT16 |
      | 伪量化 | BFLOAT16| INT8 (ND)       | FLOAT/null   | null   | null       | BFLOAT16       | BFLOAT16        | null          | INT64     | null            | null                 | null                  | BFLOAT16|
      | 伪量化 | FLOAT16 | INT4 (ND)       | FLOAT16/null | null   | null       | FLOAT16        | FLOAT16         | null          | INT64     | null            | null                 | null                  | FLOAT16 |
      | 伪量化 | BFLOAT16| INT4 (ND)       | FLOAT/null   | null   | null       | BFLOAT16       | BFLOAT16        | null          | INT64     | null            | null                 | null                  | BFLOAT16|
      | 伪量化 | INT8    | INT4 (ND/NZ)    | FLOAT        | UINT64 | null       | null           | null            | FLOAT         | INT64     | null            | null                 | null                  | BFLOAT16|
      | 伪量化 | INT8    | INT4 (ND/NZ)    | FLOAT        | UINT64 | FLOAT/null | null           | null            | FLOAT         | INT64     | null            | null                 | null                  | FLOAT16 |
      | 量化   | INT8    | INT8 (ND)       | INT32/null   | UINT64 | null       | null           | null            | null          | INT64     | null            | null                 | null                  | INT8    |
      | 量化   | INT8    | INT8 (ND)       | INT32/null   |BFLOAT16| null       | null           | null            | FLOAT/null    | INT64     | null            | null                 | null                  | BFLOAT16|
      | 量化   | INT8    | INT8 (ND)       | INT32/null   | FLOAT  | null       | null           | null            | FLOAT/null    | INT64     | null            | null                 | null                  | FLOAT16 |
      | 量化   | INT8    | INT8 (ND/NZ)    | INT32/null   | null   | null       | null           | null            | null          | INT64     | null            | null                 | null                  | INT32   |
      | 量化   | INT8    | INT8 (NZ)       | INT32/null   |BFLOAT16| null       | null           | null            | FLOAT/null    | INT64     | null            | null                 | null                  | BFLOAT16|
      | 量化   | INT8    | INT8 (NZ)       | INT32/null   | FLOAT  | null       | null           | null            | FLOAT/null    | INT64     | null            | null                 | null                  | FLOAT16 |
      | 量化   | INT4    | INT4 (ND/NZ)    | null         | UINT64 | null       | null           | null            | FLOAT/null    | INT64     | null            | null                 | null               | FLOAT16/BFLOAT16|

    - x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当x不转置时$x_i$的K轴或当x转置时$x_i$的M轴。$weight_i$的最后一维指当weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。
    - x和weight若需要转置，转置对应的tensor必须非连续。
    - 伪量化场景shape约束：
      - 伪量化场景下，若weight的类型为INT8，仅支持perchannel模式；若weight的类型为INT4，对称量化支持perchannel和pergroup两种模式。若为pergroup，pergroup数G或$G_i$必须要能整除对应的$k_i$。若weight为多tensor，定义pergroup长度$s_i = k_i / G_i$，要求所有$s_i(i=1,2,...g)$都相等。非对称量化支持perchannel模式。
      - 伪量化场景下若weight的类型为INT4，则weight中每一组tensor的最后一维大小都应是偶数。$weight_i$的最后一维指weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。并且在pergroup场景下，当weight转置时，要求pergroup长度$s_i$是偶数。
      - 伪量化参数antiquantScaleOptional和antiquantOffsetOptional的shape要满足下表（其中g为matmul组数，G为pergroup数，$G_i$为第i个tensor的pergroup数）：
          | 使用场景 | 子场景 | shape限制 |
          |:---------:|:-------:| :-------|
          | 伪量化perchannel | weight单 | $[g, n]$|
          | 伪量化perchannel | weight多 | $[n_i]$|
          | 伪量化pergroup | weight单 | $[g, G, n]$|
          | 伪量化pergroup | weight多 | $[G_i, n_i]$|
      - x为INT8、weight为INT4场景支持对称量化和非对称量化：
        - 对称量化场景：
          - 该场景下输出out的dtype为BFLOAT16或FLOAT16。
          - 该场景下offsetOptional为空。
          - 该场景下仅支持count模式（算子不会检查groupListType的值），k要求为quantGroupSize的整数倍，且要求k <= 18432。其中quantGroupSize为k方向上pergroup量化长度，当前支持quantGroupSize=256。
          - 该场景下scale为pergroup与perchannel离线融合后的结果，shape要求为$[e, quantGroupNum, n]$，其中$quantGroupNum=k \div quantGroupSize$。
          - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[e, n]$。
          - 该场景下要求n为8的整数倍。
        - 非对称量化场景：
          - 该场景下输出out的dtype为FLOAT16。
          - 该场景下仅支持count模式（算子不会检查groupListType的值）。
          - 该场景下{k, n}要求为{7168, 4096}或者{2048, 7168}。
          - scale为pergroup与perchannel离线融合后的结果，shape要求为$[e, 1, n]$。
          - 该场景下offsetOptional不为空。非对称量化offsetOptional为计算过程中离线计算辅助结果，即$antiquantOffset \times scale$，shape要求为$[e, 1, n]$，dtype为FLOAT32。
          - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[e, n]$。
          - 该场景下要求n为8的整数倍。
    - 量化场景下，若weight的类型为INT4，需满足以下约束（其中g为matmul组数，G为k轴被pergroup划分后的组数）：
      - weight的数据格式为ND时，要求n为8的整数倍。
      - 支持perchannel和pergroup量化。perchannel场景的scale的shape需为$[g, n]$，pergroup场景需为$[g, G, n]$。
      - pergroup场景下，$G$必须要能整除$k$，且$k/G$需为偶数。
      - 该场景仅支持groupType=0(x,weight,y均为单tensor)，actType=0，groupListType=0/1。
      - 该场景不支持weight转置。

    - 仅量化场景 (per-token)、反量化场景支持激活函数计算。

    - <a id="groupType-constraints"></a>不同groupType支持场景：
      - 伪量化仅支持groupType为-1和0场景。
      - 量化仅支持groupType为0场景。
      - x、weight、y的输入类型为aclTensorList，表示一个aclTensor类型的数组对象。下面表格支持场景用“单”表示由一个aclTensor组成的aclTensorList，“多”表示由多个aclTensor组成的aclTensorList。例如“单多单”，分别表示x为单tensor、weight为多tensor、y为单tensor。

      | groupType | 支持场景 | splitItem| groupListOptional | 转置 | 其余场景限制 |
      |:---------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | 多多多 | 0/1 | groupListOptional必须传空 | 1）x不支持转置<br>2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一| 1）x中tensor要求维度一致，支持2-6维，weight中tensor需为2维，y中tensor维度和x保持一致 |
      | 0 | 单单单 | 2/3 | 1）必须传groupListOptional<br>2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维<br>3）groupListOptional第1维最大支持1024，即最多支持1024个group |1）x不支持转置<br>2）支持weight转置，A8W4与A4W4场景不支持weight转置 |1）weight中tensor需为3维，x，y中tensor需为2维|
      | 0 | 单多单 | 2/3 | 1）必须传groupListOptional<br>2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维<br>3）groupListOptional第1维最大支持128，即最多支持128个group|1）x不支持转置<br>2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一 |1）x，weight，y中tensor需为2维<br>2）weight中每个tensor的N轴必须相等 |
      | 0 | 多多单 | 2/3 | 1）groupListOptional可选<br>2）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应；当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应；当groupListType为2时，groupListOptional第二列的数值需与x中tensor的第一维一一对应<br>3）groupListOptional第1维最大支持128，即最多支持128个group |1）x不支持转置<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一| 1）x，weight，y中tensor需为2维<br>2）weight中每个tensor的N轴必须相等 |
      | 2 | 单单单 | 2/3 | 1）必须传groupListOptional<br>2）当groupListType为0时，最后一个值应小于等于x中tensor的第二维；当groupListType为1时，数值的总和与x应小于等于tensor的第二维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第二维<br>3）groupListOptional第1维最大支持1024， 即最多支持1024个group | 1）x必须转置<br>2）weight不能转置 |1）x，weight中tensor需为2维，y中tensor需为3维<br>2）bias必须传空|
      | 2 | 单多多 | 0/1 | groupListOptional必须传空 | 1）x必须转置<br>2）weight不能转置<br>| 1）x，weight，y中tensor需为2维<br>2）weight长度最大支持128，即最多支持128个group<br>3）原始shape中weight每个tensor的第一维之和不应超过x第一维<br>4）bias必须传空 |

  - <term>Ascend 950PR/Ascend 950DT AI处理器</term>：
    - 非量化场景支持的数据类型为：
      - 以下入参为空：scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional、actType、activationFeatureOutOptional
      - 不为空的参数支持的数据类型组合要满足下表
        |groupType| x       | weight  | biasOptional | out     |
        |:-------:|:-------:|:-------:| :------      |:------ |
        |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
        |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
        |-1/0/2   |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|
    - 伪量化场景支持的数据类型为：
      - 以下入参为空：scaleOptional、offsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional
      - 不为空的参数支持的数据类型组合要满足下表
        |groupType| x       | weight  | biasOptional |antiquantScaleOptional|antiquantOffsetOptional| out     |
        |:-------:|:-------:|:-------:| :------      |:------|:------|:------|
        |-1/0   |BFLOAT16     |INT8/INT4     |BFLOAT16/FLOAT32/null| BFLOAT16 | BFLOAT16/null | BFLOAT16 |
        |-1/0   |FLOAT16     |INT8/INT4     |FLOAT16/null    | FLOAT16 | FLOAT16/null | FLOAT16 |
        |0   |BFLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |BFLOAT16/FLOAT32/null| BFLOAT16 | null | BFLOAT16 |
        |0   |FLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8    |FLOAT16/null    | FLOAT16 | null | FLOAT16 |
      - 当weight的数据类型为FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8时，antiquantOffsetOptional仅支持传入空指针或空tensorList，weight仅支持转置。
      - 若weight的类型为INT4，则weight中每一组tensor的最后一维大小都应是偶数。$weight_i$的最后一维指weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。
      - antiquantScaleOptional和非空的biasOptional、antiquantOffsetOptional要满足下表（其中g为matmul组数即分组数）：
        |groupType| 使用场景 | shape限制 |
        |:---------:|:---------:| :------ |
        |-1|weight多tensor|每个tensor 1维，shape为（$n_i$），不允许存在一个tensorList中部分tensor的shape为（$n_i$）部分tensor为空的情况|
        |0|weight单tensor|每个tensor 2维，shape为（g, N）|

    - 不同groupType支持场景:
      - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，out，例如单多单表示支持x为单tensor，weight多tensor，out单tensor的场景。
        | groupType | 支持场景 | 场景限制 |
        |:---------:|:-------:| :------ |
        | -1 | 多多多 |1）仅支持splitItem为0/1<br>2）非量化x，out中tensor需为2维， shape分别为（$m_i$, $k_i$）和（$m_i$, $n_i$）；伪量化场景x中tensor要求维度一致，支持2-6维，y中tensor维度和x保持一致；weight中tensor需为2维，shape为（$n_i$, $k_i$）或（$k_i$, $n_i$）；bias中tensor需为1维，shape为（$n_i$）<br>3） groupListOptional必须传空<br>4）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>5）x不支持转置<br>6）仅支持ND进ND出<br>|
        | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，shape为（g, N, K）或（g, K, N）；x，out中tensor需为2维，shape分别为（M, K）和（M, N）<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持x不转置，weight转置、不转置均支持<br>6）仅支持ND进ND出<br>|
        | 0 | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional， 且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总与x中tensor的第一维相等，长度最大为1024<br>3）x，out中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）<br>4）weight中每个tensor的N轴必须相等<br>5）支持weight转置，但weight的tensorList中每tensor是否转置需保持统一<br>6）x不支持转置<br>7）仅支持非量化<br>8）仅支持ND进ND出<br> |
        | 0 | 多多单 |1）仅支持splitItem为2/3<br>2）x，out中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N） <br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应，当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应，且长度最大为1024<br>5）支weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置  <br>7）仅支持非量化<br>8）仅支持ND进ND出<br> |
        | 2 | 单单单 |1）仅支持splitItem为2/3<br>2）x，weight中tensor需为2维，shape分别为（K, M）和（K, N）；out中tensor需为3维, shape为（g, M, N）<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）仅支持x转置且weight不转置<br>6）仅支持非量化<br>7）仅支持ND进ND出|

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_matmul](examples/test_aclnn_grouped_matmul.cpp) | 通过接口方式调用[GroupedMatmul](docs/aclnnGroupedMatmulV5.md)算子。 |