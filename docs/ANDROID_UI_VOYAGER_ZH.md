# AndroidWorld：反馈校准 SPECTRUM 与 UI-Voyager 对照

> **2026-11-01 更新：** 当前新增tau衰减与gated两组，使用
> [四组运行指南](ANDROID_VARIANTS_ZH.md)和
> [新配置](../configs/android/ui_voyager_qwen4b_variants.json)。
> 下文双组流程和82项CPU测试是原版本记录；新版本尚待Local测试与原生执行。
> 模型、设备、作者源码和manifest准备步骤继续适用。

本分支只做 **Qwen3-VL-4B-Instruct + AndroidWorld** 一组实验。
我们的递归阶段只优化反馈蒸馏 KL，不含 GRSD 或 GRPO。
对照使用 UI-Voyager 作者发布的 GRSD 分叉点算法。两个方法共享同一个
三轮 RFT 初始化，之后各跑三轮，每个方法一条训练轨迹，无温度/seed 扫描。

这是可运行的独立实现与受控比较，**不是作者论文 81% 数值的原样复现**。
LoRA、训练规模、动作长度、轮次和部分环境包装是本分支明确采用的设置。
历史代码、数学/QA/ARC 实验、数据和已有结果不受影响。

## 已验证与仍需验证

- CPU 上验证真实 tiny Qwen3-VL 的视觉前向、LoRA 梯度、冻结教师、模型合并和重载。
- CPU 上验证固定版本作者 GRSD 算法、数据隔离、谱算子恢复和训练预算。
- 最新复查与 82 项测试结果见 [复查记录](ANDROID_REVIEW_20261029.md)，
  初次验证和原生 registry 准备结果见 [验证记录](ANDROID_VALIDATION_20261028.md)。
- **尚未运行真实 4B 模型、CUDA 训练或 Android emulator**。不把代码交付当作实验成功。
- 默认单 GPU、BF16、microbatch 1、LoRA r16；4090 24GB 的实际峰值必须通过
  `qualify` 测量。没有声称单卡一定装得下最长上下文，也没有静默切到量化。

## 默认实验

| 项目 | 配置 |
|---|---|
| 模型 | Qwen/Qwen3-VL-4B-Instruct，固定 HF revision |
| 训练 | 按固定顺序选 16 个可参数化任务模板，各 2 个训练实例 |
| 固定参考 | 同样 16 个模板，各 1 个独立参数实例；原生采集 4 次 |
| 测试 | 固定版本注册表的全部 116 个任务，各 1 个独立参数实例 |
| 初始化 | 共享 3 轮成功轨迹 RFT |
| 对照 | `ui_voyager`：每轮原生采集 4 次/实例 → 作者 GRSD → LoRA |
| 我们 | `feedback_spectrum`：每轮原生 2 次 + 谱调制 2 次/实例 → 反馈 KL → LoRA |
| 递归 | 两组各 3 轮，保存并评测每轮原生 checkpoint |
| 更新预算 | 每阶段 32 optimizer updates，累积 8 个 microbatch；实际 token 数另记 |
| 解码 | temperature 0.7，top-p 0.8，top-k 20，每步最多 256 新 token |
| 图像/历史 | 当前真实截图，最多 30 步文字操作历史；图像最多 262144 pixels |
| 动作上限 | 每个 episode 最多 30 步；所有方法一致 |

这是第一组有界训练实验，不是复刻作者 7000 多训练任务。测试包含训练模板的新参数
和未参与训练的其他模板；不能将整套评测说成全部模板均未见过。
训练、参考、测试实例的参数身份严格分离，固定参数任务保留测试、排除训练选择。
任务选择、实际注册表数和排除原因均进入 private manifest。
原生 MarkorCreateFolder 的参数含墙上时钟时间，因此相同 seed 不保证重新生成的
manifest 完全相同；始终复用已经保存的 manifest，不重新生成来代替原实例。

两组匹配每轮 rollout 尝试次数和可用优化器更新次数；提前终止、回复长度和空监督
会使实际动作数/token 数不同。我们的额外参考采集、教师前向和谱校准有成本，
**不能声称总 FLOPs 或总环境交互完全相等**。参考信息只供谱校准，当前两组也不是
严格额外监督量匹配的消融；后续若做强机制结论，需要参考 replay 等信息对照。

## 1. 单独 checkout 与环境

```bash
git clone --branch android-feedback-spectrum https://github.com/Yunbo-max/SPECTRUM.git SPECTRUM-android
cd SPECTRUM-android
conda create -n spectrum-android python=3.11 pip -y
conda activate spectrum-android
python -m pip install -r requirements/android.txt
python -m pip install -e . --no-deps
python scripts/run_android_study.py fetch --source-only
python -m pip install --no-deps --no-build-isolation -e data/android/UI-Voyager/androidworld
```

按实际驱动安装匹配的 PyTorch CUDA wheel。这里使用独立环境，**不要安装旧的
`.[train]` extra**，它锁定的 Transformers 不支持本实验。
`fetch --source-only` 会跳过作者 665MB 的轨迹 LFS 文件；该文件不是本实验必需输入。
作者源码固定在 `67b65e2be093753ecaa2964f48739339b870813e`，已有不同版本目录会拒绝使用。

## 2. Android emulator 和模型

按固定版本 [原生安装说明](https://github.com/ui-voyager/UI-Voyager/blob/67b65e2be093753ecaa2964f48739339b870813e/androidworld/README.md#installation)
创建 **Pixel 6 / Android 13 Tiramisu API 33 / AndroidWorldAvd**，安装 ffmpeg。
本分支使用本地 Transformers 模型，不需要 OpenAI API key 或外部教师服务。

先修改 `configs/android/ui_voyager_qwen4b.json` 中的 SDK、adb、emulator 和 AVD 路径。
保持正式训练前配置确定。启动 emulator，例如：

```bash
~/android/emulator/emulator -avd AndroidWorldAvd -no-snapshot -port 5554 -grpc 8554
```

另一个终端，在本仓库与环境中运行：

```bash
python scripts/run_android_study.py setup-env
python scripts/run_android_study.py fetch
python scripts/run_android_study.py prepare
python scripts/run_android_study.py plan
```

`setup-env` 先下载并校验原生 accessibility forwarder APK，安装到配置指定的
emulator，再调用原生初始化安装任务 app 和权限。固定版本的 UI-Voyager 关闭了
该 APK 的自动安装，因此全新 AVD 需要这一步；普通采集不会重新安装。
`fetch` 下载一次固定版本模型并记录文件 SHA256，运行前按记录重新校验文件内容。
`prepare` 生成私有固定实例，不启动模型；既有 manifest 不覆盖。
正式运行和资格检查都先验证 manifest 内容哈希、上游版本和跨 split 实例唯一性。
没有 API 调用费用，但模型约 4.4B 参数，需预留下载、多个完整合并 checkpoint 与截图空间。
默认保留全部 checkpoint；建议至少预留约 150GB，再按实际产生的数据调整。

## 3. 原生资格检查与正式运行

先检查 GPU/ADB 可用，然后在空闲 GPU 上运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_android_study.py qualify
```

该命令只用训练/参考实例：真实采集、视觉校准、临时谱调制和恢复、一次 SFT 更新、
一次反馈更新、合并后重载。它在独立 `qualification/` 下保存记录，不纳入正式分数。
若少量探测没有得到成功经验，输出 `signal_shortage.json`，不是捏造标签或软件 PASS。
这时可先运行共享 warm-up 检查初始模型能力；正式实验中的参考采集发生在 warm-up 后。
探测成功也不证明最坏长度一定可容纳，查看记录的 peak CUDA memory 和真实长样本。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_android_study.py run --arm both
python scripts/run_android_study.py report
```

同一个命令按顺序完成共享 RFT、UI-Voyager 对照和我们的方法，默认不调度其他 GPU。
也可以分开调度（同一 output root 不允许并发拥有者）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_android_study.py warmup
CUDA_VISIBLE_DEVICES=0 python scripts/run_android_study.py run --arm ui_voyager
CUDA_VISIBLE_DEVICES=0 python scripts/run_android_study.py run --arm feedback_spectrum
```

完成的阶段按 receipt 重用。源码、配置、实例或初始 checkpoint 身份改变会拒绝旧 root。
完成记录及其引用的截图/checkpoint 哈希均检查；不要移动或覆盖原文件。
中途失败保留 partial 与 raw，不自动重复执行可能已有副作用的动作。
检查日志后，把**完整的未完成阶段目录**移到独立备份位置，再执行同一命令，才能明确重试该阶段。
不要删除或覆盖已完成轮次；不要将基础设施故障计作任务失败。

## 方法定义与边界

原生学生仅接收部署时可见的任务、截图、历史。冻结的输入 checkpoint 作为自教师，
额外读取同一任务实例的已验证成功路径或实际动作错误信息。
成功路径保留模型原始响应及其 0–999 坐标；执行器转换后的像素坐标不作为模型动作反馈。
教师与学生使用同一条原生采样 continuation，在不同长度 prompt 后精确对齐。
损失只有 response-token reverse KL，教师 detached；默认 exact-vocabulary KL，
可选 top-k union+tail 是明确标注的粗粒化 KL。

每轮从当前原生模型重算固定参考 NLL 与近期反馈 KL 的 K/V 梯度二阶矩，
各自 trace 归一化、按 alpha 混合，再用现有 SPECTRUM 近端算子。
首轮没有反馈，使用参考几何；接近零的反馈矩阵回退。仅采集阶段暂时折入原生
language K/V，训练与最终评测之前精确恢复。每轮新 LoRA，合并成一个原生学生。

这是固定采集 batch 上的反馈蒸馏，后续 optimizer updates 会重复使用前缀；
不称为每一步严格重新采样的 on-policy SDPO。谱调制轨迹只供教师参考，
不假装是原生策略采样，也不进入 GRPO likelihood ratio。

我们保留全部原始输出，包括错误动作；训练视图有明确的成功/反馈筛选，
因此这不是原始 SPECTRUM 的“全部 raw completion SFT”协议。
成功轨迹经过 native verifier 不代表其所有步骤或教师软目标都正确。
没有有信息量的反馈时跳过该训练样本；整轮无样本时记录 `no_update`，
评测未更新模型。没有真实固定参考成功时，仅谱方法被阻断。

## 对照与作者代码的差异

- baseline 调用作者发布代码的 backward state matching、`shortest_base` 和 SSIM 0.95；
  公开代码与论文中的匹配叙述有差异，以固定源代码为该对照的契约。
- baseline 先共享三轮成功 RFT，后每轮 GRSD；三轮 GRSD refinement 是我们的递归比较设置。
- 两组共同使用原生 task verifier，要求明确终止且 native score=1；partial score 另存。
- 非法动作保留错误，不偷偷替换成 wait；Menu 无对应原生操作时拒绝。
- 坐标 999 映射到最后有效像素；wait/long_press 使用原生固定时长语义。
- adapter 只训练语言注意力 Q/K/V/O，视觉编码器冻结；参考梯度仍经过真实视觉前向。

## 输出与结果解释

输出在 `runs/android/ui_voyager_qwen4b/`：

- `study.json`：冻结配置、实例/source/初始权重身份。
- `shared/`：共享 RFT、初始化评测、独立参考采集。
- 每组 `round_XX/`：raw trajectory、真实截图、监督来源、校准矩阵、训练与评测 receipt。
- `report.json`：各轮成功率、缺失状态、交互/token/更新次数及成本。

主指标是每个固定测试实例一次原生 rollout 的成功率；一条训练轨迹不支持
“跨训练随机种子稳定”结论。记录的 exact action sequence 数只是精确序列代理，
不等于语义有效路线多样性；没有实现 matched-correct 轨迹丰富度时不作该主张。
比较论文表格数值需注明本分支的训练、解码、环境和评测随机化差异。

代码入口：[study.py](../src/improving/android/study.py)、[训练](../src/improving/android/training.py)、
[视觉模型](../src/improving/android/model.py)、[几何](../src/improving/android/geometry.py)、
[原生环境](../src/improving/android/environment.py)、[作者 GRSD 接口](../src/improving/android/baseline.py)。

## 来源

- [UI-Voyager paper](https://arxiv.org/abs/2603.24533)，[作者 MIT 源码](https://github.com/ui-voyager/UI-Voyager/tree/67b65e2be093753ecaa2964f48739339b870813e)。原 prompt 运行时读取，不在此复制。
- [SDPO](https://arxiv.org/abs/2601.20802)：反馈条件化自教师和逐 token 分布蒸馏思想。
- 本仓库既有 SPECTRUM：固定参考梯度几何与生成阶段近端谱调制。
- 反馈参与下一轮几何是本分支提出的实验机制；未声称已证实有效或完成创新查重。
