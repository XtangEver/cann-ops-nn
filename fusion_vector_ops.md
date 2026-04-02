# ops-nn 仓库 vector+vector 融合算子梳理

## 概述

ops-nn 仓库的 vector+vector 融合算子集中在 **`vfusion/`** 目录下，共包含 **6 个算子**（含反向）：

| # | 算子 | 路径 | 说明 |
|---|------|------|------|
| 1 | **modulate** | `vfusion/modulate/` | y = x * (scale + 1) + shift |
| 2 | **modulate_grad** | `vfusion/modulate_grad/` | modulate 的反向 |
| 3 | **scaled_masked_softmax_v2** | `vfusion/scaled_masked_softmax_v2/` | softmax(x * scale) with mask |
| 4 | **scaled_masked_softmax_grad_v2** | `vfusion/scaled_masked_softmax_grad_v2/` | 上述算子的反向 |
| 5 | **multi_scale_deformable_attn_function** | `vfusion/multi_scale_deformable_attn_function/` | 多尺度可变位注意力前向 |
| 6 | **multi_scale_deformable_attention_grad** | `vfusion/multi_scale_deformable_attention_grad/` | 上述算子的反向 |

### 架构

```
vfusion/
├── modulate/                    # element-wise: y = x*(scale+1) + shift
│   ├── op_kernel/
│   │   ├── modulate.cpp         # 主 kernel (template-based for B/L/D tiling)
│   │   ├── modulate_apt.cpp     # APT (Auto Parallel Tiling?) 版本
│   │   └── arch35/              # Regbase 架构特定实现
│   │       ├── modulate_regbase.h
│   │       ├── modulate_regbase_common.h
│   │       ├── modulate_regbase_tiling_key.h
│   │       └── modulate_struct.h
│   ├── op_host/
│   │   ├── modulate_tiling.cpp  # Tiling 策略实现
│   │   ├── modulate_tiling.h    # Tiling 数据结构
│   │   ├── modulate_regbase_tiling.cpp
│   │   ├── modulate_def.cpp     # 算子定义
│   │   └── op_api/              # ACL API 封装
│   ├── docs/aclnnModulate.md    # 文档
│   └── tests/                   # 单元测试
├── modulate_grad/               # modulate 反向
├── scaled_masked_softmax_v2/    # ScaledMaskedSoftmaxV2
├── scaled_masked_softmax_grad_v2/
├── multi_scale_deformable_attn_function/
└── multi_scale_deformable_attention_grad/
```

### 统一调试方式

所有 vfusion 算子遵循相同的构建和调试流程：

**1. 编译：** 使用仓库根目录的 `build.sh`，指定算子名称和 SOC 版本：
```bash
bash build.sh --ascend910b
```

**2. Binary 编译（预编译 kernel）：**
```bash
# 生成 binary 编译任务
bash scripts/kernel/binary_script/build_binary_opc_gen_task.sh <op_type> <soc> <output_path>
# 编译
bash scripts/kernel/binary_script/build_binary_opc.sh <op_type> <soc> <output_path>
```

**3. 单元测试运行：**
```bash
# 测试 host 侧 tiling
cd tests/ut/op_host && ./test_aclnn_<op>.cpp
# 测试 kernel 侧
cd tests/ut/op_kernel && ./test_<op>.cpp
```

**4. Tiling 数据打印调试：** 所有算子的 tiling 代码中都包含 `TilingDataPrint()` 或 `OP_LOGD` 调用，在 DEBUG 模式下会输出所有 tiling 参数：
```cpp
// 示例：modulate_tiling.cpp
OP_LOGD(context, "inputB: %ld.", tilingData.inputB);
OP_LOGD(context, "ubLength: %ld.", tilingData.ubLength);
```

**5. 示例程序：** 每个算子都有 `examples/test_aclnn_<op>.cpp` 作为最小可运行的示例。

---

## 1. modulate

### 功能

融合 element-wise 的 **Scale + Add** 操作：
```
y = x * (scale + 1) + shift
```

- `scale` 和 `shift` 为**可选输入**，支持三种模式：
  - `SCALE_AND_SHIFT`（0）：同时提供 scale 和 shift
  - `NO_SCALE`（1）：只提供 shift
  - `NO_SHIFT`（2）：只提供 scale

- 输入 shape: `[B, L, D]` (batch, length, dimension)
- 支持的 dtype: `float` (DT_FLOAT), `half` (DT_FLOAT16), `bfloat16_t` (DT_BF16)

### 文件结构

| 文件 | 路径 | 说明 |
|------|------|------|
| 主 kernel | `vfusion/modulate/op_kernel/modulate.cpp` | 核心 AscendC kernel |
| Tiling 头 | `vfusion/modulate/op_host/modulate_tiling.h` | Tiling 数据结构定义 |
| Tiling 实现 | `vfusion/modulate/op_host/modulate_tiling.cpp` | Tiling 策略算法 |
| Regbase tiling | `vfusion/modulate/op_host/modulate_regbase_tiling.h` | 旧架构 Tiling |
| Arch35 kernel | `vfusion/modulate/op_kernel/arch35/modulate_regbase.h` | Arch35 专用 kernel |
| APT kernel | `vfusion/modulate/op_kernel/modulate_apt.cpp` | Auto Tiling 版本 |
| 算子定义 | `vfusion/modulate/op_host/modulate_def.cpp` | op proto 定义 |
| ACL API | `vfusion/modulate/op_host/op_api/aclnn_modulate.cpp` | ACL 接口封装 |

### 核心代码逻辑 (modulate.cpp)

#### 基类 `ModulateBase<T>`

所有三种 Tiling 策略的 kernel 共享同一个基类 `ModulateBase<T>`，提供以下基础设施：

```cpp
template <typename T>
class ModulateBase {
protected:
    // Buffer: QueueX 和 QueueY 使用双缓冲 (DOUBLE_BUFFER=2)
    // QueueScale 和 QueueShift 使用单缓冲 (1)
    TPipe* pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> QueueX;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> QueueY;
    TQue<QuePosition::VECIN, 1> QueueScale;
    TQue<QuePosition::VECIN, 1> QueueShift;
};
```

#### 三大计算路径

1. **Compute** (同时有 scale 和 shift)：
   ```cpp
   // 对于 bfloat16：cast to fp32 → Mul(x, scale+1) → Add(result, shift) → cast back
   // 对于 fp16/fp32：Mul(xLocal, scaleLocal) → Add(result, shiftLocal)
   ```

2. **ComputeWithoutScale** (只有 shift)：
   ```cpp
   Add(yLocal, xLocal, shiftLocal)  // y = x + shift
   ```

3. **ComputeWithoutShift** (只有 scale)：
   ```cpp
   Mul(yLocal, xLocal, scaleLocal)  // y = x * (scale + 1)
   ```

#### 数据搬运

- `CopyIn/CopyOut`：使用 `DataCopyPad` 实现从 GM 到 Local 的搬运，支持 padding 和对齐
- bfloat16 特殊处理：存储时偏移半个 ubLength 以优化对齐

#### 入口函数

```cpp
extern "C" __global__ __aicore__ void modulate(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        ModulateB<DTYPE_X> op(&pipe, &tilingData); // 按 B 维度分块
        op.Init(x, scale, shift, y); op.Process();
    } else if (TILING_KEY_IS(1)) {
        ModulateL<DTYPE_X> op(&pipe, &tilingData); // 按 L 维度分块
        op.Init(x, scale, shift, y); op.Process();
    } else if (TILING_KEY_IS(2)) {
        ModulateD<DTYPE_X> op(&pipe, &tilingData); // 按 D 维度分块
        op.Init(x, scale, shift, y); op.Process();
    }
}
```

### Tiling 逻辑 (modulate_tiling.cpp)

#### TilingData 结构

```cpp
struct TilingDataStructModulate {
    int64_t inputB, inputL, inputD;  // 输入三维度
    int64_t ubLength;                 // UB 可用长度（每个数据元素）
    int64_t frontNum, frontLength;   // 前 frontNum 个 core，每个处理 frontLength 个元素
    int64_t tailNum, tailLength;     // 剩余 tailNum 个 core，每个处理 tailLength 个元素
    int64_t useDTiling;             // 是否使用 D 维度分块
    int64_t parameterStatus;        // 0=SCALE_AND_SHIFT, 1=NO_SCALE, 2=NO_SHIFT
};
```

#### 策略选择 (`SelectStrategy()`)

根据输入 shape 和 core 数量自动选择最优分块维度：

```
if (B * L < coreNum):
    // 总任务太少，检查 D 维度是否足够大
    if (B * L * (D/coreNum) * dtype_size >= 32KB):  -> TilingD
    else:                                            -> TilingB
elif (B >= coreNum):                                -> TilingB  // 最理想：直接按 batch 分
elif (B >= coreNum / 2):
    if (D * (B*L/coreNum) * dtype_size >= 32KB):    -> TilingL
    else:                                            -> TilingB
else:                                                -> TilingL
```

#### Tiling 参数计算 (`CalcTilingParam()`)

```cpp
tailLength = totalElements / coreNum;        // 每核基础处理量
frontLength = tailLength + 1;               // 前几个核多处理 1 个元素
frontNum = totalElements % coreNum;         // 前 frontNum 个核多处理
tailNum = (tailLength == 0) ? 0 : coreNum - frontNum;
```

这种分块方式确保负载均衡，每个 core 要么处理 N 个元素，要么处理 N+1 个元素。

#### UB 容量计算

```cpp
// tensor 数量：(parameterStatus == SCALE_AND_SHIFT) ? 6 : 5
// 6 tensor: x, y, scale, shift + double buffer for x(2) + double buffer for y(2) = 实际上 2+2+1+1=6
// 5 tensor: 缺少 scale 或 shift 时
ubLength = ubSizePlatform / tensorNum / dtypeSize / 64 * 64  // 64 字节对齐
```

### 调试要点

1. **bfloat16 padding**：bfloat16 需要特殊的 `ubLength / 2` 偏移处理，在 `DataCopyPad` 时传入偏移
2. **参数状态检测**：在 Init 阶段根据 `scaleShape` 和 `shiftShape` 是否为空来判断 parameterStatus
3. **Tiling 日志**：编译时加 `-DDEBUG` 可通过 `OP_LOGD` 观察完整的 Tiling 数据输出

---

## 2. modulate_grad

### 功能

modulate 的反向传播。根据前向公式 `y = x * (scale + 1) + shift`：

```
grad_input   = grad_output * (scale + 1)   (如果有 scale)
             = grad_output                   (如果没有 scale)
grad_scale   = grad_output * x              (如果有 scale，需在 L 维度上求和)
grad_shift   = grad_output                   (如果有 shift，需在 L 维度上求和)
```

### 文件结构

| 文件 | 路径 | 说明 |
|------|------|------|
| 主 kernel | `vfusion/modulate_grad/op_kernel/modulate_grad.cpp` | 核心 AscendC kernel |
| Tiling | `vfusion/modulate_grad/op_host/modulate_grad_tiling.cpp/.h` | Tiling 策略 |
| ACL API | `vfusion/modulate_grad/op_host/op_api/aclnn_modulate_backward.cpp` | ACL 接口 |
| 文档 | `vfusion/modulate_grad/REAMDE.md` | README（注意文件名拼写为 REAMDE） |

### 核心代码逻辑 (modulate_grad.cpp)

#### 主 kernel：`ModulateGradKernel<T>`

- 输入: `input`, `grad_output`, `scale`
- 输出: `grad_input`, `grad_scale`, `grad_shift`
- 固定使用 `float32` 计算

```cpp
template<typename T>
class ModulateGradKernel {
    // 9 个 Queue（比前向复杂得多）
    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECIN, 1> gradOutputQueue_;
    TQue<TPosition::VECIN, 1> scaleQueue_;
    TQue<TPosition::VECOUT, 1> gradInputQueue_;
    TQue<TPosition::VECOUT, 1> gradShiftLocalQueue_;
    TQue<TPosition::VECOUT, 1> gradScaleLocalQueue_;
    // 累加队列
    TQue<TPosition::VECIN, 1> tileGradShiftQueue_;
    TQue<TPosition::VECIN, 1> tileGradScaleQueue_;
    TQue<TPosition::VECIN, 1> currenttileGradScaleQueue_;
};
```

#### 计算流程 (ComputeGradients)

```
for each row l in currentL_:
    if has_shift_:
        tileGradShift += gradOutput[l]    // 沿 L 维度求和
    if has_scale_:
        gradInput[l] = gradOutput[l] * scale    // dL/dx
        currenttileGradScale = input[l] * gradOutput[l]  // dL/ds 的部分和
        tileGradScale += currenttileGradScale
    else:
        gradInput[l] = gradOutput[l]    // 无 scale 时直接拷贝

// 跨 tile 累加
gradScale += tileGradScale
gradShift += tileGradShift
```

### Tiling 逻辑

#### 分块策略

1. **splitByD (按 D 分块)**：
   - 用于 D > MAX_TILE_LENGTH / sizeof(T) 的场景
   - `coresPerB = totalCoreNum / B`（每个 batch 分配若干 core）
   - 每个 core 处理 D 的一段连续区域

2. **splitByB (按 B 分块)**：
   - 用于 D 在 UB 内可容纳的场景
   - D 不分块，直接在 B 维度上分配 core

3. **L 维度内分块**：
   - 当 `isDOverflow` (D 太大) 时，在 L 维度上逐行处理
   - `maxRowsPerTile = MAX_TILE_LENGTH / totalD_`
   - 否则一次处理尽可能多的 L 行

#### Tiling 数据结构

```cpp
struct ModulateGradTiling {
    uint32_t B, L, D;
    uint32_t has_scale, has_shift;
    uint32_t splitB;          // 0=按B分, 非零=按D分
    uint32_t coresPerB;
    uint32_t usedCores;
    uint32_t formerNum, formerLength, tailNum, tailLength;  // 标准负载均衡
};
```

### 调试要点

1. **UB 容量限制**：`UB_CAPACITY = 192KB`, `MAX_TILE_LENGTH = 12KB / sizeof(T)`（对 float 为 3072）
2. **32 字节对齐**：`alignedRowBytes` 确保每次 DataCopy 是 32 字节对齐的
3. **梯度累加**：grad_scale 和 grad_shift 需要在 L 维度上求和，使用 `tileGradShift/tileGradScale` 作为中间累加 buffer

---

## 3. scaled_masked_softmax_v2

### 功能

将 softmax 与 mask + scale 融合为一个 kernel：
```
scaled_x = x * scale
masked_x = select(scaled_x, mask, MASK_VAL=-10000.0)  // mask=True 的位置替换为 -10000
y = softmax(masked_x, dim=-1)
```

- 输入 x: `[batch, channel, height, width]` — 在 width 维度做 softmax
- 输入 mask: `[maskBatch, maskChannel, height, width]` — 支持 broadcasting
- 属性 `scale`: float

### 文件结构

| 文件 | 路径 | 说明 |
|------|------|------|
| Kernel 头 | `vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.h` | 完整 kernel 实现（~320 行） |
| Kernel 入口 | `vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.cpp` | dtype dispatch |
| Tiling 头 | `vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_tiling.h` | Tiling 数据结构 |
| Tiling 实现 | `vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_tiling.cpp` | Tiling 策略 |

### 核心代码逻辑 (scaled_masked_softmax_v2.h)

```cpp
template <typename T>
class ScaledMaskedSoftmaxV2 {
    // 三个 Queue
    TQue<QuePosition::VECIN, 1> inQueueX;         // x 输入
    TQue<QuePosition::VECIN, 1> inQueueMask;      // mask 输入 (bool)
    TQue<QuePosition::VECOUT, 1> outQueueY;        // 输出
    LocalTensor<float> scaledMaskedX;             // 中间 buffer (fp32)
    LocalTensor<uint8_t> sharedBuffer;            // Softmax 共享 buffer
};
```

#### 计算流程

1. **CopyIn**: 同时搬运 x 和 mask 到 local
   - mask 搬运需要处理跨 batch/channel 边界的情况（`CalcCurPos`/`CalcEndPos`）
   
2. **Compute**:
   ```
   if dtype == bfloat16:
       Cast(scaledMaskedX, xTensor) -> fp32
       Muls(scaledMaskedX, scaledMaskedX, scale)
       SelectWithBytesMask(scaledMaskedX, scaledMaskedX, -10000, maskTensor)
       SoftmaxX(scaledMaskedX, scaledMaskedX)   // 就地
       Cast(yTensor, scaledMaskedX) -> bf16
   elif dtype == half:
       Muls(xTensor, xTensor, scale)
       SelectWithBytesMask(xTensor, xTensor, -10000, maskTensor)
       Cast(scaledMaskedX, xTensor) -> fp32
       SoftmaxX(scaledMaskedX, scaledMaskedX)
       Cast(yTensor, scaledMaskedX) -> fp16
   else (float):
       Muls(xTensor, xTensor, scale)
       AscendC::SelectWithBytesMask(scaledMaskedX, xTensor, MASK_VAL, maskTensor)  // 跨类型
       SoftmaxX(yTensor, scaledMaskedX)  // 直接输出
   ```

3. **CopyOut**: 将结果搬回 GM

#### Softmax 高阶 API

使用 `SoftMax<float, false, false>()` 高阶 API，需要传入：
- `SoftMaxTiling` — 通过 `AscendC::SoftMaxTilingFunc()` 生成
- `SoftMaxShapeInfo` — `{lines, padLineNum, lines, width}`

### Tiling 逻辑 (scaled_masked_softmax_v2_tiling.cpp)

#### TilingData 结构

```cpp
struct ScaledMaskedSoftmaxV2TilingData {
    uint64_t coreNum;              // AIV core 数量
    uint64_t batch, channel, height, width;  // x shape
    uint64_t maskBatch, maskChannel, maskHeight, maskWidth;  // mask shape
    float scale;
    uint64_t maskMode;             // broadcast 模式
    uint64_t paddingNum, padLineNum;         // x padding 信息
    uint64_t alignedMaskPadding, alignedMaskWidth;  // mask 对齐
    uint64_t nStep, cStep;         // mask 跨 batch/channel 映射
    uint64_t headCoreNum;          // 大核（处理多行的核）数量
    uint64_t lineHeadCore, iterHeadCore, lineHeadIter, lineLastHeadIter;  // 大核参数
    uint64_t lineTailCore, iterTailCore, lineTailIter, lineLastTailIter;  // 小核参数
    SoftMaxTiling softmaxTilingData;  // softmax 高阶 API 专用 tiling
};
```

#### 核心分配策略

```
totalLine = batch * channel * height   // 总行数
lineTailCore = totalLine / coreNum     // 尾行核每核处理行数
headCoreNum = totalLine % coreNum      // 大核数量（多处理一行）
lineHeadCore = lineTailCore + 1        // 大核处理行数

// 对大核/小核分别计算：
iterHeadCore = CeilDiv(lineHeadCore, availableLinePerIter)   // 循环次数
lineHeadIter = availableLinePerIter                          // 每次循环行数
lineLastHeadIter = lineHeadCore - (iterHeadCore - 1) * lineHeadIter  // 尾循环行数
// 小核同理
```

#### UB 分配

```
maxByteLine = padLineNum * 2 * dtypeSize + padLineNum * 4 + maskPaddedWidth * 1
//              x + y 的 padded 行                      fp32 buffer          mask padded 行

availableLinePerIter = (availableUbSize - softmaxBuffSize) / maxByteLine
// softmaxBuffSize = 32KB 或 64KB（Regbase 平台）
```

#### Mask 广播适配

```cpp
// maskMode 位掩码:
// bit 0 (BROADCAST_BATCH): batch != maskBatch 时需要广播
// bit 1 (BROADCAST_CHANNEL): channel != maskChannel 时需要广播

if (batch != maskBatch):
    nStep = 0  // mask 不沿 batch broadcast
else:
    nStep = 1

if (channel != maskChannel):
    cStep = 0
else:
    cStep = 1
```

### 调试要点

1. **TilingKey 计算**: 由 dtype 决定
   - `FP32: key=0`
   - `FP16: key=1`
   - `BF16: key=2`
   
2. **数据宽度限制**: `width` 必须在 `(0, 4096]` 范围内（Regbase 平台为 8192）

3. **Mask 搬运复杂分支**：当 mask 跨 batch/channel 时，`CopyMaskIn` 有 7 个分支，需要特别注意
   - 同一 batch + 同一 channel
   - 同一 batch + 不同 channel
   - 不同 batch + 跨 channel 首尾 + 中间

4. **Softmax tmp buffer**：共享 buffer 大小由 `GetSoftMaxMaxTmpSize()` 决定，限制在 32KB/64KB

---

## 4. scaled_masked_softmax_grad_v2

### 功能

`scaled_masked_softmax_v2` 的反向传播。

公式: `grad_input = grad_output * (softmax_output - sum(grad_output * softmax_output)) * scale`

### 文件结构

| 文件 | 路径 | 说明 |
|------|------|------|
| Kernel | `vfusion/scaled_masked_softmax_grad_v2/op_kernel/` | kernel 实现 |
| Tiling | `vfusion/scaled_masked_softmax_grad_v2/op_host/` | tiling 策略 |

### 调试要点

- 与正向算子类似，Tiling 策略也按 batch/channel/height 分片
- 需要注意 softmax 反向中需要前向的 softmax 输出作为额外输入

---

## 5. multi_scale_deformable_attn_function

### 功能

多尺度可变位注意力 (Multi-Scale Deformable Attention):
```
output = sum over (level, point): attention_weights[level, point] * 
         bilinear_interp(value[level], sampling_locations[level, point])
```

### 架构选择

```
if ARCH 310P (ascend310p):
    使用 ms_deform_attn_310p.h (InfBase 架构)
else (910b, 910_93, 950 等):
    使用 ms_deform_attn_high_perf.h (高性能版本)
    或 ms_deform_attn_generic.h (通用版本)
```

### 文件结构

| 文件 | 路径 | 说明 |
|------|------|------|
| Kernel 入口 | `vfusion/multi_scale_deformable_attn_function/op_kernel/multi_scale_deformable_attn_function.cpp` | dtype/arch dispatch |
| Kernel 头 | `vfusion/multi_scale_deformable_attn_function/op_kernel/ms_deform_attn_310p.h` | 310P 架构实现 |
| 高性能 kernel | `vfusion/multi_scale_deformable_attn_function/op_kernel/ms_deform_attn_high_perf.h` | 高性能优化版本 |
| 通用 kernel | `vfusion/multi_scale_deformable_attn_function/op_kernel/ms_deform_attn_generic.h` | 通用实现 |
| Tiling | `vfusion/multi_scale_deformable_attn_function/op_host/multi_scale_deformable_attn_function_tiling.cpp` | Tiling 策略 |
| Infershape | `vfusion/multi_scale_deformable_attn_function/op_host/multi_scale_deformable_attn_function_infershape.cpp` | 输出形状推导 |

### 核心代码逻辑 (op_kernel 入口)

```cpp
extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function(...) {
#if __CCE_AICORE__ == 200  // ascend310p
    if (TILING_KEY_IS(1)) {
        KernelMultiScaleDeformableAttn310P<float> op;  // 专用 310P 实现
        op.Init(...); op.MSDAProcess();
    }
#else  // 910b/910_93/950 等
    if (TILING_KEY_IS(1002)) { KernelMultiScaleDeformableAttnOpt<2, 16> op; ... }  // 2 points, 16 embed
    if (TILING_KEY_IS(1004)) { KernelMultiScaleDeformableAttnOpt<4, 16> op; ... }  // 4 points, 16 embed
    if (TILING_KEY_IS(1008)) { KernelMultiScaleDeformableAttnOpt<8, 16> op; ... }  // 8 points, 16 embed
    if (TILING_KEY_IS(2002)) { KernelMultiScaleDeformableAttnOpt<2, 32> op; ... }  // 2 points, 32 embed
    if (TILING_KEY_IS(2004)) { KernelMultiScaleDeformableAttnOpt<4, 32> op; ... }  // 4 points, 32 embed
    if (TILING_KEY_IS(2008)) { KernelMultiScaleDeformableAttnOpt<8, 32> op; ... }  // 8 points, 32 embed
    if (TILING_KEY_IS(0))    { KernelMultiScaleDeformableAttn op;        ... }  // 通用路径
}
```

### Tiling 逻辑

#### TilingKey 编码

TilingKey 由 `embedDims` 和 `numPoints` 编码：

```cpp
// optPoint 条件：numLevels <= 8 && numHeads <= 8 && (embedDims==16||32) && (numPoints % 2 == 0)
if (optPoint):
    groups = GroupPoints(numPoints)     // 将 points 分组：{8, N/8} | {4, N/4} | {2, N/2} | {1, N}
    TilingKey = (embedDims / 16) * 1000 + pointGroupSize     // 如 1002 = 1*1000 + 2
else:
    TilingKey = 0  // 通用路径
```

#### 输入约束

| 参数 | 约束 |
|------|------|
| numQueries | >= 32 |
| numHeads | <= 16 |
| numLevels | <= 16 |
| numPoints | <= 16 |
| embedDims | 8 的倍数，且 <= 256 |

### 调试要点

1. **Deterministic 模式**：当 `deterministicFlag == 1` 时，`coreNum = 1`（强制单核）
2. **模板特化**：高性能版本使用 `<numPoints, embedDims>` 模板参数实现编译期优化
3. **sysWorkspaceSize**：固定使用 16MB workspace

---

## 6. multi_scale_deformable_attention_grad

### 功能

`multi_scale_deformable_attn_function` 的反向传播，计算三个梯度：
- `grad_value` - value 的梯度
- `grad_samplingLoc` - 采样位置的梯度
- `grad_attnWeight` - 注意力权重的梯度

### 文件结构

| 文件 | 路径 | 说明 |
|------|------|------|
| Kernel 入口 | `vfusion/multi_scale_deformable_attention_grad/op_kernel/multi_scale_deformable_attention_grad.cpp` | 主 kernel |
| Kernel 头 | `vfusion/multi_scale_deformable_attention_grad/op_kernel/multi_scale_deformable_attention_grad.h` | 类定义 |
| Tiling | `vfusion/multi_scale_deformable_attention_grad/op_host/multi_scale_deformable_attention_grad_tiling.cpp` | Tiling |
| Infershape | `vfusion/multi_scale_deformable_attention_grad/op_host/multi_scale_deformable_attention_grad_infershape.cpp` | 输出形状推导 |

### 调试要点

- 与 forward 类似，支持多种 TilingKey 对应不同模板实例化
- 反向需要额外的 value 作为输入来计算 gradient

---

## 附：Matmul Epilogue Fusion（算子内融合）

除 `vfusion/` 目录外，还有 matmul epilogue 融合组件，这些是 **内嵌于 matmul kernel 的 vector+vector 融合模块**：

### ops-nn matmul epilogue fusion

路径: `ops-nn/matmul/common/cmct/epilogue/fusion/`

| 文件 | 功能 | 说明 |
|------|------|------|
| `default_fusion_op.h` | 默认融合（空操作） | 当不启用融合时，直接 `dstLocal = srcLocal` |
| `fusion_add.h` | 融合加法 (Matmul + Add) | 将 matmul 输出与第三个 tensor 相加: `output = matmul(A,B) + C` |
| `fusion_gelu.h` | 融合 GELU (Matmul + GELU) | 对 matmul 输出应用 GELU 激活 |
| `fusion_mul.h` | 融合乘法 (Matmul + Mul) | 将 matmul 输出与第三个 tensor 逐元素相乘 |

#### `FusionAdd` 核心逻辑

```cpp
class FusionAdd {
    // 1. 从 GM 搬运第三个输入到 Local
    DataCopyPad(inputLocal_, inputGlobal_[offset], copyParams, padParams);
    // 2. 等待数据就绪
    WaitFlag<MTE2_V>(ZERO_FLAG);
    // 3. vector 融合: output = src + input
    Add(outputLocal, inputLocal_, srcLocal, stageSize);
    // 4. 同步
    PipeBarrier<PIPE_V>();
};
```

### ops-transformer group matmul epilogue fusion

路径: `ops-transformer/gmm/common/cgmct/epilogue/fusion/`

与 ops-nn 的 matmul epilogue 结构相同，用于 grouped matmul 的 epilogue 融合。
