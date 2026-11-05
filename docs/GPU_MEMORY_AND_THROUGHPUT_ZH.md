# 显存参数与 GPU 吞吐：AndroidWorld / MobileGym

更新日期：2026-11-05。模型为 Qwen3-VL-4B-Instruct，BF16、原生视觉输入、LoRA 反馈蒸馏。本文区分专项短测与完整实验；完整训练和评测仍在进行，不能将短测当作最终成功率。

## 当前采用的配置

| 项目 | AndroidWorld | MobileGym |
|---|---|---|
| 物理 GPU | GPU 1 | GPU 1 |
| 单进程 PyTorch allocator 上限 | 40 GiB | 40 GiB |
| 模型副本 | 1 | 1 |
| 独立环境并发 | 1 个 Android 模拟器 | 4 个独立浏览器环境，共享模型 |
| 训练 microbatch | 4 | 4 |
| 梯度累积 | 2 | 2 |
| 每次参数更新的有效 batch | 8 | 8 |
| 完整评测任务数 | 116 | 256 |

两任务 allocator 上限合计 80 GiB。上限不是预分配，也不是进程全部显存的精确硬上限：CUDA context、驱动及其他进程开销还需额外余量。`CUDA_VISIBLE_DEVICES` 限定物理 GPU 1 后，进程内部使用逻辑 `cuda:0` 是正常的。稳定绑定时可使用 `nvidia-smi -L` 查到的 GPU UUID。

原有两套配置的默认训练参数均是 batch 1、累积 8。新提供的 [四环境配置](../configs/mobilegym/feedback_only_qwen4b_parallel4.json) 是显式选择，不覆盖原默认配置。

### 主机实测与配置审阅（2026-11-05 22:25–22:29 UTC）

原生 Windows 检测得到 AMD Ryzen Threadripper PRO 9955WX、16 个物理核心、32 个逻辑线程、255.38 GiB 系统内存。一次 3 秒采样的全机 CPU 使用率为 33.4%，可用 RAM 为 174.54 GiB；这只是采样窗口，不是长期峰值。两张卡各报告 97887 MiB，实验仅使用 GPU 1。该时段 GPU 1 的设备占用约 31–32 GiB，会随阶段变化。`emulator -accel-check` 确认 WHPX 已安装并可用。

这些数据支持继续观察当前并行运行，但不能证明并行比顺序执行更快。按 CPU 线程数直接划分串行/并行的表格只能作为起始经验，不能代替两任务的配对吞吐测量。目前保留 MobileGym 4 环境、AndroidWorld 1 个 AVD。

**AndroidWorld 的 batch 4 是当前封存运行配置，不是“完整长上下文资格测试已通过”的结论。** 上表 batch 2/4 的既有训练峰值来自 MobileGym 原生训练反馈。AndroidWorld 自身的长轨迹、教师反馈和接近 8192-token 的显存验收仍需单独记录。保守的新运行可先用 batch 2 / accumulation 4，再测试 batch 4 / accumulation 2；不能原地修改已经封存的实验来混合两种设置。有效 batch 相同保持了每次更新的样本数，但浮点计算、分组和随机采样仍可能不同，不保证逐位相同结果。

Windows 的 `scripts/windows/run_benchmark.ps1 -Benchmark MobileGym -Full` 仍选择原始单环境配置，**不会自动选择四环境优化**。明确选择新配置的命令如下（在仓库根目录执行，模型和服务需已准备好）：

```powershell
$env:CUDA_VISIBLE_DEVICES = "1"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
& .\.venv-guiagent\Scripts\python.exe scripts\run_mobilegym_feedback.py `
  --config configs/mobilegym/feedback_only_qwen4b_parallel4.json run
```

这个命令不会附加到已运行的实验，也不会自行设置 40 GiB allocator 上限；已有任务在跑时不要重复启动。当前主机使用独立封存配置和本地调度器，继续运行不依赖此通用 `-Full` 入口。CPU 线程 4 是限制线程争用的起点，不是该 CPU 的已证明最优值。

内存限制的 API 语义核对使用本机对应的 [PyTorch 2.8 官方文档](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.set_per_process_memory_fraction.html)。模拟器 VM 加速与 GPU 图形加速的区别见 [Android 官方文档](https://developer.android.com/studio/run/emulator-acceleration)。本地 83 passed / 1 skipped 不应被描述为已经核实此提交的远端 Windows CI 通过。

## 哪些参数影响多少 GB？

这里的 GiB = 2^30 字节；十进制 GB = 10^9 字节，两者不要混算。系统 RAM、GPU 显存、磁盘模型文件大小也不是同一个指标。

| 参数 | 当前值 | 增大后的主要影响 | 调整时需要保留的约束 |
|---|---:|---|---|
| `train.batch_size` | 4 | 同时保留更多样本的激活、输出和反向图，通常提高训练峰值；不一定严格线性 | 与累积共同决定有效 batch，先做真实前向/反向测试 |
| `train.gradient_accumulation_steps` | 2 | 分多次处理样本后更新；不会像同时增大 microbatch 一样复制整批激活 | 有效 batch = microbatch × 累积；此处为单训练进程 |
| `model.max_context_tokens` | 8192 | 只是准入上限；实际较长输入会增加激活、注意力和推理 KV cache | 当前实现超限报错，不会悄悄截断历史 |
| `model.max_pixels` | 262144 | 允许更大截图，通常增加视觉 token、计算和显存 | 不是固定每张图都用这么多像素；需与原生 processor 尺寸规则一起看 |
| `model.min_pixels` | 65536 | 提高最小图像预算可能增加小图的视觉 token | 改分辨率可能改变任务表现，不能只比较速度 |
| `generation.max_new_tokens` | 256 | 实际生成越长，推理 KV cache、耗时越多 | 达到上限可能导致动作截断；正式实验不为提速而缩短预算 |
| `study.history_len` | 30 | 更长的实际文字历史增加上下文 | 这是过去动作历史长度；不等同于环境并发数 |
| `model.dtype` | `bfloat16` | 改成 FP32 会增大权重和许多张量；总峰值不一定恰好翻倍 | 当前路径要求原生 BF16/FP32，不能直接替换成量化模型 |
| `train.gradient_checkpointing` | `true` | 开启可用额外重算换取更少保存的训练激活 | 主要影响反向训练，不会自动加快环境交互 |
| `train.lora_rank` | 16 | 更高 rank 增加可训练参数、梯度和优化器状态 | 会改变学习容量，不能只作为显存开关 |
| `train.logit_chunk_size` | 16 | 影响 KL 临时张量与循环开销 | exact KL 的语义保持不变；峰值需实测 |
| `mobilegym.environment_workers` | 4 | 此分支增加浏览器、CPU/RAM 和排队请求；共用一个模型且串行进入推理 | 不代表训练 batch；也不加载 N 份模型，通常不会令显存变成 N 倍 |
| 模型进程数量 | 每任务 1 | 每个进程通常有独立权重、CUDA context、KV cache | 多开模型副本与多开浏览器环境是不同方案 |
| allocator 上限 | 每任务 40 GiB | 限制 PyTorch allocator；不预先填满显存 | 显存分配失败仍可发生；增加上限本身不会让运算变快 |

在训练代码中，教师使用冻结输入检查点并禁用适配器，不单独常驻第二份教师模型。教师 logits 转到 CPU 后进行学生前向；内存需求仍与响应长度、词表张量和批次组成有关。参考 [训练实现](../src/improving/android/training.py) 和 [原生视觉模型](../src/improving/android/model.py)。

## 训练 batch 的实测峰值

真实 4B BF16、LoRA、反馈 reverse-KL，包含反向、AdamW 更新和保存。下面不是最大上下文压力测试。

| 专项配置 | 实际分配峰值 | allocator 保留峰值 |
|---|---:|---:|
| batch 2、累积 4；4 条真实记录循环使用 | 11.67 GiB | 12.35 GiB |
| batch 4、累积 2；同一组 4 条记录 | 10.19 GiB | 10.72 GiB |
| batch 8、累积 1；8 条真实记录 | 11.82 GiB | 12.71 GiB |

每项执行 3 次真实更新。batch 2 的测量高于 batch 4，说明不能用这些短测拟合简单的线性显存公式；分配缓存、微批次边界和输出张量生命周期都会影响峰值。batch 8 使用的记录集合也不同，不能当作严格配对比较。

batch 8 是独立专项实现的资格测试，学生提示约 877–996 tokens、教师约 2098–2202 tokens、响应 28–59 tokens；**不能据此保证 8192-token 训练也只需 13 GiB**。本分支正式训练入口支持 microbatch 1/2/4，当前采用 4；没有默认启用 batch 8。

`max_memory_allocated` 表示张量分配峰值，`max_memory_reserved` 包括 allocator 缓存；`nvidia-smi` 是设备/进程的另一层统计，整卡数值还包括其他工作负载。

## 1、2、4、24 环境吞吐测试

测试仅使用训练任务、真实截图、真实模型动作和原生状态判分。诊断把每条轨迹限制为 2 个动作，并使用 greedy 解码进行比较；**正式配置仍是原来的 30 动作上限和随机采样**。

| 同一轮比较 | 环境数 | 轨迹 / 实际动作 | 总耗时 | 轨迹/分钟 | allocator 保留峰值 |
|---|---:|---:|---:|---:|---:|
| A | 1 | 8 / 16 | 84.172 s | 5.70 | 8.62 GiB |
| A | 2 | 8 / 16 | 47.109 s | 10.19 | 8.62 GiB |
| A | 4 | 8 / 16 | 41.860 s | 11.47 | 8.62 GiB |
| B | 4 | 24 / 48 | 134.750 s | 10.69 | 8.62 GiB |
| B | 24 | 24 / 48 | 136.297 s | 10.57 | 8.62 GiB |

B 组实际观察到同时存在 24 个浏览器。24 环境没有显示出比 4 环境更快，约 1% 的差距也不能据此认定稳定的速度差；因此选择更省 CPU/RAM 的 4 环境。A 组短测约 2 倍提升不等于完整训练/评测也会快 2 倍。各测试共享 GPU，与其他正式任务存在调度干扰，没有重复测量置信区间。B 组整卡平均利用率约 72.5% / 70.1%，包含其他工作，不能解释成被测进程单独的利用率。

实时界面截图存在变化，少量生成坐标不同。两组任务实例、动作数量和原生分数一致；这些短轨迹的成功数均为 0，属于吞吐检查，不能宣称模型完成任务的能力已验证。另用四份固定截图提示复核，原调用与异步串行推理返回完全一致的 token；两个真实浏览器的 origin storage 隔离检查通过。

## 采用与复用规则

1. 使用独立配置和全新输出目录；不要编辑正在运行实验的封存配置或源代码。
2. `mobilegym.environment_workers` 支持 1/2/4/8/16/24。此处只资格测试过 1/2/4/24，8/16 不应被误报为实测通过。模型推理始终由一个线程串行执行，环境的等待和动作可以重叠。
3. [四环境配置](../configs/mobilegym/feedback_only_qwen4b_parallel4.json) 可交给现有 `scripts/run_mobilegym_feedback.py --config ... run` 入口；先按 Windows 安装指南准备模型、Chromium 和 MobileGym 服务。该 JSON **本身不会设置 allocator 上限**；当前本地调度器另行在加载模型前调用下面的 PyTorch API。

```python
# 在导入 torch 之前绑定 CUDA_VISIBLE_DEVICES 到选定 GPU；此进程只看到一张卡。
import torch
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(40 * 2**30 / total, device=0)
```

4. 显式复用通过 `reuse_completed_episodes.config_path` 和 `config_sha256` 指向原封存配置，且仅允许输出目录和环境并发数变化。检查源 study seal、任务、策略、split、检查点、原生判分和截图；只复用完整 episode，不把 partial 当作失败或成功。
5. 本次切换复用了 128 条 Round 0 和 36 条已完成的第一轮原生采集轨迹。后续新检查点不复用旧检查点的轨迹；不支持链式 A→B→C 复用。原始截图路径继续被引用，因此必须保留原输出目录。
6. 并发改变新请求进入采样器的顺序；这不是原进程 RNG 状态的逐位续跑。任务规模、原生判分、反馈来源、训练更新数与有效 batch 均保持原协议。每轮完成后才报告完整评测成绩。

测试覆盖调度无丢失/重复、取消时完成清理、模型入口串行、配置与源数据复用拒绝条件、微批次梯度等。故障会留下 partial 记录并中止，不伪造完成结果。私有清单、原始任务轨迹、截图和模型文件不包含在本次 GitHub 更新中。
