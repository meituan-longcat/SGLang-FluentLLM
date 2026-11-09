# aclnnGroupedMatmul

**须知：该接口后续版本会废弃，请使用最新aclnnGroupedMatmulV5接口。**

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

- 接口功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。根据x、weight、y的Tensor数量支持如下4种场景：

  - x、weight、y都为多tensor，即每组的数据对应的Tensor是独立的。
  - x为单tensor，weight/y为多tensor，此时需要通过可选参数group_list说明x在行上的分组情况，如group_list[0]=10说明x的前10行参与第一组矩阵乘计算。
  - x、weight为多tensor，y为单tensor，此时每组矩阵乘的结果放在同一个Tensor中连续存放。
  - x、y为单tensor，weight为多tensor，属于前两种情况的组合。

    **说明：** 单tensor指一个tensor list中所有分组的tensor在M轴上合并为1个；否则为多tensor。
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

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedMatmulGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmul”接口执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(
    const aclTensorList  *x,
    const aclTensorList  *weight,
    const aclTensorList  *biasOptional,
    const aclTensorList  *scaleOptional,
    const aclTensorList  *offsetOptional,
    const aclTensorList  *antiquantScaleOptional,
    const aclTensorList  *antiquantOffsetOptional,
    const aclIntArray    *groupListOptional,
    int64_t               splitItem,
    const aclTensorList  *y,
    uint64_t             *workspaceSize,
    aclOpExecutor        **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmul(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnGroupedMatmulGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed;width: 1540px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 118px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th><a href="../../../docs/zh/context/数据格式.md" target="_blank">数据格式</a></th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th><a href="../../../docs/zh/context/非连续的Tensor.md" target="_blank">非连续的Tensor</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>公式中的输入x。</td>
        <td>
          <ul>
            <li>支持的最大长度为128个。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、INT8</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>公式中的weight。</td>
        <td>
          <ul>
            <li>支持的最大长度为128个。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、INT8</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>可选输入</td>
        <td>公式中的bias。</td>
        <td>
          <ul>
            <li>长度与weight相同。</li>
          </ul>
        </td>
        <td>FLOAT16、FLOAT32、INT32、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleOptional</td>
        <td>可选输入</td>
        <td>代表量化参数中的缩放因子。</td>
        <td>
          <ul>
            <li>长度与weight相同。</li>
          </ul>
        </td>
        <td>UINT64</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetOptional</td>
        <td>可选输入</td>
        <td>代表量化参数中的偏移量。</td>
        <td>
          <ul>
            <li>长度与weight相同。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScaleOptional</td>
        <td>可选输入</td>
        <td>代表伪量化参数中的缩放因子。</td>
        <td>
          <ul>
            <li>长度与weight相同。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>可选输入</td>
        <td>代表伪量化参数中的偏移量。</td>
        <td>
          <ul>
            <li>长度与weight相同。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListOptional</td>
        <td>可选输入</td>
        <td>代表输入和输出M方向的matmul索引情况。</td>
        <td>
          <ul>
            <li>长度与weight相同。</li>
            <li>当输出中TensorList的长度为1时，groupListOptional中的最后一个值约束了输出数据的有效部分，groupListOptional中未指定的部分将不会参与更新。</li>
          </ul>
        </td>
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
          <ul>
            <li>0/1代表输出为多tensor；2/3代表输出为单tensor。</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>公式中的输出y。</td>
        <td>
          <ul>
            <li>支持的最大长度为128个。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、FLOAT32</td>
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
    <tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - x、weight支持FLOAT16、BFLOAT16、INT8
    - y支持FLOAT16、BFLOAT16、INT8、FLOAT32
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - x支持FLOAT16、BFLOAT16、FLOAT32
    - weight支持FLOAT16、BFLOAT16、FLOAT32、INT8
    - y支持FLOAT16、BFLOAT16、FLOAT32
    - 不支持scaleOptional、offsetOptional

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景时报错:

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>必选输入、输出或者必选属性是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、splitItem、y的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>weight的长度不在支持范围。</td>
    </tr>
    <tr>
      <td>若bias不为空，bias的长度不等于weight的长度。</td>
    </tr>
    <tr>
      <td>splitItem为0、1的场景，y长度不等于weight的长度，groupListOptional长度不等于weight的长度。</td>
    </tr>
    <tr>
      <td>splitItem为2、3的场景，y长度不等于1。</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmul

- **参数说明：**
    <table>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulGetWorkspaceSize获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnGroupedMatmul默认确定性实现。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 非量化场景支持的输入类型为：
    - x为FLOAT16、weight为FLOAT16、biasOptional为FLOAT16、scaleOptional为 空、offsetOptional为空、antiquantScaleOptional为空、 antiquantOffsetOptional为空、y为FLOAT16；
    - x为BFLOAT16、weight为BFLOAT16、biasOptional为FLOAT32、scaleOptional 为空、offsetOptional为空、antiquantScaleOptional为空、 antiquantOffsetOptional为空、y为BFLOAT16；
  - 量化场景支持的输入类型为：

    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为UINT64、 offsetOptional为空、antiquantScaleOptional为空、 antiquantOffsetOptional为空、y为INT8；
  - 伪量化场景支持的输入类型为：
    - x为FLOAT16、weight为INT8、biasOptional为FLOAT16、scaleOptional为空，  offsetOptional为空，antiquantScaleOptional为FLOAT16、 antiquantOffsetOptional为FLOAT16、y为FLOAT16；
    - x为BFLOAT16、weight为INT8、biasOptional为FLOAT32、scaleOptional为 空，offsetOptional为空，antiquantScaleOptional为BFLOAT16、 antiquantOffsetOptional为BFLOAT16、y为BFLOAT16；
  - 如果传入groupListOptional，groupListOptional必须为非负递增数列，groupListOptional长度不能为1。
  - 当前支持的场景：
      支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，y，例，单多单表示支持x为单tensor，weight多tensor，y单tensor的场景。

      | 支持场景 | 场景限制 |
      |:-------:| :-------|
      | 多多多 |1）仅支持splitItem为0/1<br>2）x中tensor要求维度一致，支持2-6维，weight中  tensor需为2维，y中tensor维度和x保持一致<br>3）若x中存在tensor大于2维，  groupListOptional必须传空<br>4）若x中tensor为2维且传入  groupListOptional，groupListOptional的差值需与x中tensor的第一维一一对应 |
      | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，且最后 一个值与x中tensor的第一维相等<br>3）x,weight,y中tensor需为2维<br>4） weight中每个tensor的N轴必须相等 |
      | 单多多 |1）仅支持splitItem为0/1<br>2）必须传groupListOptional， groupListOptional的差值需与y中tensor的第一维一一对应<br>3）x,weight,y中  tensor需为2维 |
      | 多多单 |1）仅支持splitItem为2/3<br>2）x,weight,y中tensor需为2维 <br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional， groupListOptional的差值需与x中tensor的第一维一一对应 |

  - x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当属 性transpose_x为false时$x_i$的K轴或当transpose_x为true时$x_i$的M轴。  $weight_i$的最后一维指当属性transpose_weight为false时$weight_i$的N轴或当  transpose_weight为true时$weight_i$的K轴。
  - x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值  2147483647。

- <term>Ascend 950PR/Ascend 950DT</term>：
    <details>
    <summary>非量化场景约束</summary>
      <a id="非量化场景约束"></a>

  - 非量化场景支持的数据类型为：
    - 如果传入groupListOptional，groupListOptional必须为非负递增数列，groupListOptional长度不能为1
    - 以下入参为空：scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional
    - 不为空的参数支持的数据类型组合要满足下表：
      | x       | weight  | biasOptional | y     |
      |:-------:|:-------:| :------      |:------ |
      |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
      |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|

    </details>

    <details>
    <summary>伪量化场景约束</summary>
      <a id="伪量化场景约束"></a>

    - 伪量化场景支持的数据类型为：
      - 以下入参为空：scaleOptional、offsetOptional
      - 不为空的参数支持的数据类型组合要满足下表：
          | x       | weight  | biasOptional | antiquantScaleOptional | antiquantOffsetOptional | y     |
          |:-------:|:-------:| :------      |:------ |:------ |:------ |
          |BFLOAT16    |INT8     |BFLOAT16/FLOAT32/null    | BFLOAT16 | BFLOAT16 | BFLOAT16 |
          |FLOAT16     |INT8     |FLOAT16/null             | FLOAT16  | FLOAT16  | FLOAT16  |
      - antiquantScaleOptional和非空的biasOptional、antiquantOffsetOptional要满足下表：

        | 使用场景 | shape限制 |
        |:---------:| :------ |
        |weight多tensor|每个tensor 1维，shape为（$n_i$），不允许存在一个tensorList中部分tensor的shape为（$n_i$）部分tensor为空的情况 |
    - 仅支持多多多场景。
    </details>

    <details>
    <summary>支持场景约束</summary>
      <a id="支持场景约束"></a>

      - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x、weight、y。例如单多单表示支持x为单tensor、weight多tensor、y单tensor的场景。

        | 支持场景 | 场景限制 |
        |:-------:| :-------|
        | 多多多 |1）仅支持splitItem为0/1<br>2）伪量化场景x中tensor要求维度一致，支持2-6维，y中tensor维度和x保持一致；非量化场景x，y中tensor需为2维， shape分别为（$m_i$, $k_i$）和（$m_i$, $n_i$）；weight中tensor需为2维，shape为（$n_i$, $k_i$）或（$k_i$, $n_i$）；bias中tensor需为1维，shape为（$n_i$）<br>3）若x中存在tensor大于2维，groupListOptional必须传空<br>4）若x中tensor为2维且传入  groupListOptional，groupListOptional的差值需与x中tensor的第一维一一对应，且长度最大为128<br>5）仅支持ND进ND出<br>6）不支持x转置，不支持weight转置 |
        | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，最后一个值与x中tensor的第一维相等，且长度最大为128<br>3）x，y中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>4） weight中每个tensor的N轴必须相等<br>5）仅支持ND进ND出<br>6）仅支持非量化<br>7）不支持x转置，支持weight转置（weight若为多tensor，则每个tensor是否转置须保持一致）|
        | 单多多 |1）仅支持splitItem为0/1<br>2）必须传groupListOptional，最后一个值与x中tensor的第一维相等，且长度最大为128<br>3）x，y中tensor需为2维，shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>4）仅支持ND进ND出<br>5）仅支持非量化<br>6）不支持x转置，支持weight转置（weight若为多tensor，则每个tensor是否转置须保持一致）|
        | 多多单 |1）仅支持splitItem为2/3<br>2）x，y中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N） <br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional， groupListOptional的差值需与x中tensor的第一维一一对应，且长度最大为128<br>5）仅支持ND进ND出<br>6）仅支持非量化<br>7）不支持x转置，支持weight转置（weight若为多tensor，则每个tensor是否转置须保持一致）|

    </details>

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul.h"

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
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    std::vector<T> hostData(GetShapeSize(shape), 0);
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
    std::vector<std::vector<int64_t>> xShape = {{1, 16}, {4, 32}};
    std::vector<std::vector<int64_t>> weightShape= {{16, 24}, {32, 16}};
    std::vector<std::vector<int64_t>> biasShape = {{24}, {16}};
    std::vector<std::vector<int64_t>> yShape = {{1, 24}, {4, 16}};
    void* xDeviceAddr[2];
    void* weightDeviceAddr[2];
    void* biasDeviceAddr[2];
    void* yDeviceAddr[2];
    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclIntArray* groupedList = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* y = nullptr;
    int64_t splitItem = 0;

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

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmul第一段接口
    ret = aclnnGroupedMatmulGetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmul第二段接口
    ret = aclnnGroupedMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    for (int i = 0; i < 2; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %hu\n", j, resultData[j]);
        }
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    for (int i = 0; i < 2; i++) {
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
