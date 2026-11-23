# Qwen3.5-0.8B：共享主干的规划头与 GUI decision head

本轮入口：[交接记录](../rounds/20261123-qwen08-dualhead/WEB_HANDOFF.md) ·
[配置](../configs/gui_decision/qwen35_08b_2060.json) ·
[源码](../src/improving/gui_decision/) · [CLI](../scripts/gui_decision.py)。

状态：**generated_unexecuted**。已编写完整数据、特征、训练、双头推理和评测入口；
Web 只做源码/静态检查，未加载模型、运行项目测试、训练或评测。
RTX 2060 的具体显存、驱动、系统和执行路径仍由 Local 从实际主机恢复。
本文件是这一独立任务的 Local guide；原 AndroidWorld/MobileGym/SPECTRUM 任务不使用此环境。

## 1. 方法与边界

保留一个 `Qwen3_5ForConditionalGeneration`，原 `lm_head` 负责 PLAN；
新增共享候选 scorer 负责 ACT。两条路径引用同一份 `.model` 主干权重。
ACT 直接调用 base model 的 `last_hidden_state`，不计算整个语言词表 logits。
PLAN 使用原 `generate()`，greedy、关闭 thinking、最多 64 个新 tokens。

| 控制 prompt | 输出 | 使用方式 |
|---|---|---|
| `<GUI_MODE:PLAN>` | 下一步短子目标文本 | 原语言头；没有可执行动作，不单独计动作成功率 |
| `<GUI_MODE:ACT>` | 候选动作得分与选中动作 | 新 decision head；使用调用者提供的 instruction |
| `<GUI_MODE:PLAN_ACT>` | 自生成子目标 + 选中动作 | PLAN 的输出成为 ACT 的 instruction |

当前 gate 是**显式 prompt 路由**，由调用层解析可信控制参数。
它不是训练出来的难度判别器，也不是自动决定是否重规划的策略。
页面文字和 a11y 候选中的标记不能改变这个路由。
两个 head 的输出空间分别是语言词表和当前候选集合，不能直接相加 logits。

`PLAN_ACT` 共享参数，但会先规划，再用新 instruction 重建 ACT prompt，
因此有两次多模态 prefill 和规划阶段的自回归解码；没有实现跨两阶段的视觉特征复用。
“一次截图前向”仅适用于一次 ACT 的特征提取。

候选 scorer 读取：最后有效 token 的 query、每条候选描述末尾的隐藏状态、
该控件框对应的合并后图像 token ROI、几何/可访问性 flags、动作 type embedding。
共享投影及 MLP 为每个候选产生一个 scalar，按有效 mask 取 argmax。
不同屏幕候选数量可变；这是候选评分，不是固定 80 类分类或生成坐标文本。
候选文本在 causal prompt 中有顺序，因此不声称端到端对候选重排严格等变。

第一阶段只训练 decision head：原视觉/语言主干及原 LM head 全冻结，
单目标 CE、非 click 样本权重 1.5（记录的 long_press 仍按原规则加权）。
原规划头没有通过本轮训练获得新的 GUI 规划能力，其实际效果必须评测。
没有加入 OPSD、RL、LoRA、JEPA latent predictor 或 learned gate。
这是现有接口迁移及双头基线，**不把加一个 head 或显式路由声明为已验证的新论文贡献**。

## 2. 数据与评测口径

使用官方 **AndroidControl** 的截图、a11y tree、goal、step instruction、记录动作；
不需要 AndroidWorld、MobileGym、模拟器或在线奖励。
官方 split JSON 为 train 13,603 / validation 137 / test 1,543 episodes；
实际导出步骤数、异常及候选覆盖由 manifest 记录，不硬填论文里的有效样本数。

每屏幕最多保留 80 个 a11y 元素：可见、可操作、合法相交框，按面积升序并去重。
每个元素有 click，editable 另有 input_text，外加四方向 scroll、back、home、wait、open_app。
动作数为 M + E + 8，最多 168。点击位置取候选原框中心。
GT 不参与枚举、排序或补充候选；无法匹配候选的训练样本跳过 loss 并计数，评测仍计错。

| 条件 | 模型可见 instruction | 用途 |
|---|---|---|
| ACT Low | 当前数据集 step instruction | 检查 decision head 在已知子目标下的选择能力 |
| ACT High | 任务 goal | 同一个 head 的直接动作决策对照 |
| PLAN_ACT High | PLAN 只看 goal；ACT 看模型生成的子目标 | 真正检验本轮双头组合 |

三者仍是 **offline single-step**：每步从 released screenshot 及最多 4 步记录历史开始。
history 来源是 recorded ground truth；当前和未来动作不进入输入。
PLAN 无条件忽略 Low instruction；正式 PLAN_ACT 入口只允许 High。
这不是完整任务 rollout 成功率，也不能直接与 screenshot-only ScreenSpot/Pro 比较。

主指标为全部步骤的 native step accuracy（百分数），并保留 type accuracy、
各动作类型指标、候选覆盖、所有 GT clicks 的成功率和条件 click grounding。
后者仅对类型正确且存在 GT 框的 click 统计，不能冒充完整点击成功率。
主 judge 保留 GUI-JEPA 的 long_press→click、框内中心、scroll 方向、文本归一化等规则；
open_app 使用主评测的原始字符串精确匹配，不替换成另一 baseline 的归一化版本。

input_text 使用冻结的已发布 MiniLM span ranker；open_app 使用仅从 train 构造的词表。
两者根据**预测动作类型**补参数，再独立判分，不根据 GT 类型决定输出。
参数模块也是完整系统的一部分，不能把整体称为“只有 0.8B、没有额外模型”。
默认没有 GUI-JEPA 的 click reranker。参数枚举为显式稳定的排序/哈希采样版本；
不声称与原脚本随机遍历逐位一致，差异见 [第三方说明](../third_party/gui_jepa/README.md)。

默认完整 test；可额外使用 `paper_idd`（IDD 减所有 OOD，716 episodes）、
`official_idd`（原 IDD，721）及原始重叠 OOD 成员。
不能用 test 或 IDD 选超参。原 Qwen 直接生成坐标的完整对照尚未接入本轮 CLI，
没有它的同协议结果，不能宣称此方法优于 Qwen 原生坐标生成。

## 3. Local 环境与路径

下面命令均从本仓库根目录执行。Local 先读取本文件及交接记录的确切交付版本，
恢复实际 GPU/VRAM、驱动、CPU/RAM、磁盘、当前作业以及现有 SSH/Conda 路径。
若 GPU 在另一台机器，控制器通过现有 SSH/harness 分发这些原生 argv；
此交接不假定新的 SSH 地址、运行中的任务、远程 Codex 服务或训练授权。
对实际开始的任务记录命令、版本、stdout/stderr、耗时、显存和退出状态。

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
conda create -n gui-decision-qwen08 python=3.11 pip -y
conda activate gui-decision-qwen08
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/gui_decision.txt
python -m pip install -e . --no-deps
python -m pip freeze > gui-decision-environment.txt
```

这是候选依赖组合，需在实际系统验证 CUDA 驱动和依赖解析；本轮未实机安装。
不要在这个环境安装旧 `.[train]`、`requirements/android.txt`、FlashAttention 或 vLLM。
默认 FP16 + SDPA，head FP32；没有 BF16/FlashAttention 要求。

初始化图像预算 262,144 pixels，名义最多约 256 merged image tokens；
实际 rectangular grid 由 processor 决定并校验。候选描述前 128 字符是显式预处理；
完整候选、任务和历史不被静默删除。8192 是**输入接受上限，不是已测显存预算**。
超过上限会报错，改输入预算必须产生新配置/新 cache，并记录方法条件的改变。

## 4. Download datasets and models

确切模型 revision、GCS 对象 generation/bytes/官方 MD5 及来源 commit 在
[assets.json](../configs/gui_decision/assets.json)。下载命令不会隐式运行模型。

```bash
python scripts/gui_decision.py download --asset qwen
python scripts/gui_decision.py download --asset arguments
python scripts/gui_decision.py download --asset data
python scripts/gui_decision.py download --asset all --verify-only
python scripts/gui_decision.py prepare
```

Qwen 完整 snapshot 保存于标准 HF revision cache，默认运行 `local_files_only=true`。
span ranker 仅下载所需目录/许可证/清单到 `data/gui_decision/assets/gui_jepa`。
原始 20 个 gzip TFRecord shards 共 49,930,232,975 bytes（约 49.93 GB），另有两个 split 文件。
下载通过固定 GCS generation 读取，支持 `.partial` 范围续传并校验官方 MD5。
重复执行验证已存在输入，不删已完成 cache。HF 记录本地内容 SHA，区别于 GCS 官方 MD5。

准备输出 `data/gui_decision/androidcontrol/{train,validation,test}.jsonl`、
原 PNG 截图、train app vocabulary、split 文件及 `manifest.json`。
默认 TFRecord CRC 检查和严格失败。若官方 train 存在不可解析 episode，可显式使用
`python scripts/gui_decision.py prepare --output-dir data/gui_decision/androidcontrol_repaired --on-error quarantine_train`，
保留全部隔离 ID/原因并更新新配置路径；validation/test 损坏永远中止，不缩小分母。

请为 raw、PNG、特征 cache 分别留磁盘空间。PNG 实际体积和 cache 平均候选数未测；
每个 view 的两组 FP16 1024维候选特征约占 `N × 4096` bytes，另有 query/辅助特征/文件开销。
以 150,000 个 views、平均 40 候选估算约 24.6 GB；这是容量估算，不能当作实测。
缓存实际截图 SHA 在提取时记录，旧 GUI-JEPA 的 ViT `.pt` cache 不能复用。

## 5. Calibrate, debug, then dispatch

当前仅交付实现，不自动启动训练或大规模评测。Local 在既有授权/预算内执行，
如使用 research-autopilot，将以下 argv 纳入真实执行主机的 admitted harness task，
绑定实际 GPU/CPU/RAM/I/O 及剩余额度；不要把此文档当成已经启动的任务。

先进行软件/接口验收，使用本轮 authored tests 和真实 validation 样本：

```bash
python -m pytest tests/gui_decision -q
python scripts/gui_decision.py extract --split validation
```

第二条调用真实 0.8B、完整 validation 的 Low/High views，并输出
`data/gui_decision/qwen08_features/validation/extraction_receipt.json`，
记录实际 token 上限、耗时、GPU名、allocated/reserved 峰值及描述裁剪统计。
任何模型/图片/长度错误都使 cache 保持 incomplete，不能用于完整评测。
这不是训练效果结果；软件 fixtures 也不能替代真实模型/设备验收。

获准开始第一阶段后，完整 train、固定 validation 选 checkpoint：

```bash
python scripts/gui_decision.py extract --split train
python scripts/gui_decision.py train
```

特征提取 batch=1，训练只加载缓存与评分 head，feature batch=32、6 epochs、
AdamW 1e-4、weight decay .01。只有一次训练轨迹，不默认扫描种子/温度。
种子在 `runs/private/gui_decision_seed.json` 生成并保存，公开记录仅使用该文件的 digest。
按 validation Low/High native step accuracy 的平均值选择 `best`，每轮保存指标及 `last`。
原 planner 保持冻结；这一步不是 planner 预训练，也没有 learned gate loss。

中断恢复使用相同配置和输入：

```bash
python scripts/gui_decision.py extract --split train --resume
python scripts/gui_decision.py train --resume
```

训练恢复包含 head、optimizer、epoch/step、torch/loader/CUDA RNG。
cache 指纹、数据、参数模块或训练配置不同会拒绝恢复，使用新目录而非覆盖旧结果。

检查训练后双头接口，可先导出一个**真实 High observation**：

```bash
python scripts/gui_decision.py export-observation --mode high --output runs/gui_decision/observation.json
python scripts/gui_decision.py predict --observation runs/gui_decision/observation.json --gate-prompt '<GUI_MODE:PLAN>' --output runs/gui_decision/plan.json
python scripts/gui_decision.py predict --observation runs/gui_decision/observation.json --gate-prompt '<GUI_MODE:ACT>' --output runs/gui_decision/act.json
python scripts/gui_decision.py predict --observation runs/gui_decision/observation.json --gate-prompt '<GUI_MODE:PLAN_ACT>' --output runs/gui_decision/plan_act.json
```

这三条是接口验收，不是完整 benchmark。正式保存全部结果：

```bash
python scripts/gui_decision.py extract --split test
python scripts/gui_decision.py evaluate --split test
python scripts/gui_decision.py evaluate-online --split validation --mode high --route PLAN_ACT
python scripts/gui_decision.py evaluate-online --split test --mode high --route PLAN_ACT
```

`evaluate` 同时输出 ACT Low/High；`evaluate-online` 真正生成每步 plan，再调用同一 head。
后者不能复用预先基于 GT Low instruction 提取的缓存来冒充 self-planned 特征。
固定 test 条件后再运行；validation 的组合策略诊断不能转化为 test 调参。

| 待比较的条件 | 已实现入口 | 解释 |
|---|---|---|
| 同一 head，GT Low instruction | `evaluate` 的 low | decision 能力参照，不是自主规划 |
| 同一 head，goal-only High | `evaluate` 的 high | 直接 ACT 对照 |
| 同一 head，原 LM 自生成子目标 | `evaluate-online --route PLAN_ACT --mode high` | 双头组合 |
| 原 Qwen 直接生成坐标 | 尚未接入本轮 | 后续论文效果结论的必要强对照，当前不作优越性声明 |
| 模型自行决定何时 PLAN | 尚未实现 learned gate | 当前 prompt 路由不能替代这项实验 |

## 6. 故障定位与返回

| 现象 | 源码/证据 | 有限处理 |
|---|---|---|
| 依赖/模型 API 不符 | `requirements/gui_decision.txt`、`processor.require_transformers_version` | 恢复独立环境与确切版本，不混装旧实验依赖 |
| raw/split/CRC 错误 | `assets.py` receipt、`androidcontrol.py`、prepared manifest | 检查 generation/MD5/未完成文件；支持的续传保留部分数据 |
| image token/ROI/offset 错误 | `processor.py`、`spatial.py`、样本 ID/grid | 保留原数据定位；不能固定14×14或删除难样本 |
| OOM或长度超限 | extraction receipt、实际 GPU进程、完整失败输入 | 先排查占用；配置变化仅在 validation 校准，新 fingerprint/new cache；不删候选保分母 |
| NaN/FP16 overflow | `cache.py`、`model.extract_features` | 明确失败并保存原输入；不能把 nonfinite 静默置零 |
| train/val cache 不相容 | `runner.train`、cache manifest | 使用相同模型/processor条件重新提取，禁止伪造指纹 |
| ACT不错但PLAN_ACT变差 | 全量 prediction 中的 plan、action、GT native judge | 区分规划错误、生成指令分布变化、候选漏召；先看 validation |
| 点击分数看似很高 | `evaluation.py` 各分母与candidate coverage | 区分条件点击指标、全部GTclicks、全部步骤，不混报 |

实施性修复可在原方法范围内进行：每个已定位原因最多一次改动和一次相关复查，
失败保留记录；新的机制、learned gate、planner LoRA 或论文结论需要单独审阅设计。
本轮没有设定或启动 GPU 时间/费用预算，也没有继承旧 96GB/27B 实验的资源额度。

保留 `run_identity.json`、epoch JSON、cache/extraction manifest、原始 predictions、
metrics、执行 commit/dirty patch、实际依赖版本/资源/退出状态；失败和负结果一并返回。
把小型审查包写在本 repo 的 `rounds/20261123-qwen08-dualhead/returns/<run-id>/`，
不上传原数据、模型权重、private seed、凭据或整个运行目录。
没有新增 HF 输出仓库。读取实际结果前不报告训练成功、显存达标或准确率提升。
