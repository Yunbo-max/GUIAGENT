# AndroidWorld：接入已有 tau 与 gated 版本

日期：2026-11-01。交付分支：`android-feedback-spectrum`。
状态：**generated_unexecuted**；新代码与测试源码经过源代码审查，尚未在本轮执行项目测试、4B/CUDA或模拟器实验。旧版82项CPU测试不能作为本修订通过的证据。

## 版本来源与比较范围

本修订迁移已有代码实验中的强度控制，不新增主干或训练损失：

- 来源提交：主分支 `b8bcc68f87a2a216ac048bc7fb36365c2002e8b0`。
- tau：`configs/spectrum_7b/spectrum_7b_r10_spectral_4096_tau06.yaml`，最新已有衰减系数0.6；历史0.8配置保留在主分支。
- gated：`configs/spectrum_7b/spectrum_7b_r10_spectral_4096_adaptive.yaml`，`tau_mode=adaptive`、`tau_gamma=5`。
- GUI父版本：`eb0008b0fa64332803e75937e6df8ef07a209c9e`。

新入口是 `configs/android/ui_voyager_qwen4b_variants.json`，使用全新
`runs/android/ui_voyager_qwen4b_variants/`。原双组配置和历史结果不修改；
旧运行须继续使用其原代码版本，不能把新源码替换进已sealed的运行。

| 实验组 | CLI arm | 强度 |
|---|---|---|
| UI-Voyager GRSD | `ui_voyager` | 无谱干预 |
| 原反馈SPECTRUM | `feedback_spectrum` | 固定tau=1 |
| SPECTRUM-tau | `feedback_spectrum_tau` | tau_r = 1 × 0.6^(r−1) |
| SPECTRUM-gated | `feedback_spectrum_gated` | tau_r = 1 × (H_r/H_1)^5 |

保持Qwen3-VL-4B-Instruct与AndroidWorld、共享3轮RFT初始化、各组3轮refinement，
以及原来的实例manifest、动作预算、解码、LoRA和反馈KL设置。
三组SPECTRUM每实例每轮各采集2次原生轨迹+2次谱轨迹；GRSD仍4次原生轨迹。
成功固定参考采集在三组SPECTRUM间共享；后续反馈、checkpoint和gate参考归属各自实验组。
各轮最终评测都使用恢复后的原生模型。没有自动发起实验或后台任务。

## 数学与实现边界

算子仍是 `T_r = [I + tau_r (I − C_mix/lambda_max(C_mix))]^-1`，其中
`C_mix` 是当前checkpoint重新计算的参考NLL与近期反馈KL梯度二阶矩的trace归一化混合。
只有生成阶段临时使用该算子，随后恢复原权重，再进行原来的反馈蒸馏。

gated的H取自**当前checkpoint在固定参考上的NLL梯度二阶矩**，与主分支gate的信号来源保持一致；
它不从成功率、测试结果或反馈样本数量取信号，也不对混合反馈矩阵计算gate熵。
对每个K/V模块，取严格正特征值并按总和归一化成p，然后计算
`H_module = −sum(p log p) / log(number_of_positive_eigenvalues)`；对有效模块取算术平均。
全零或无效矩阵直接报错；其余正特征值不足两个的模块跳过。
若所有模块均不可用，adaptive显式报错，不能悄悄改成固定tau。

H_1来自**共享RFT之后，该arm的第一轮refinement**，写入带哈希的几何回执；
后续轮次与断点恢复都复用它。无效/零参考熵、非有限比率或强度均明确拒绝。
实际tau与H、H_1、比率、轮次进入几何回执和报告，不从未完成阶段猜测参考。

保持原gated公式，**没有额外截断或上限**：H_r高于H_1时tau可以大于初值。
这个熵按正特征值数量归一化，低秩但均匀的谱仍可有H=1；它不等于有效维数，
也不能保证侦测所有退化。谱熵下降导致tau减小不等于任务成功率必然提高。
本次另补充数值与恢复检查，没有声称已证明gate有效。

三轮tau衰减依次为1、0.6、0.36，不能把代码任务十轮结果直接当成GUI结果。
`tau_decay` 与 `tau_mode=adaptive` 互斥；gated不是按token学习的新门控网络，
而是根据每轮校准统计计算的标量强度。

## Local运行顺序

沿用[原Android指南](ANDROID_UI_VOYAGER_ZH.md)里的独立环境、模型/作者源码固定版本、
Android13/API33 Pixel6、真实manifest和原生验证器。已有模型与manifest可复用。
机器必须实际支持可用KVM；本修订不解决宿主机未映射 `/dev/kvm` 的问题。
仅在实际运行主机上进行下列资格检查与执行；Web没有运行这些命令：

```bash
python -m pytest tests/test_android_*.py -q
python scripts/run_android_study.py --config configs/android/ui_voyager_qwen4b_variants.json plan
python scripts/run_android_study.py --config configs/android/ui_voyager_qwen4b_variants.json qualify --arm feedback_spectrum_tau
python scripts/run_android_study.py --config configs/android/ui_voyager_qwen4b_variants.json qualify --arm feedback_spectrum_gated
python scripts/run_android_study.py --config configs/android/ui_voyager_qwen4b_variants.json run --arm all
python scripts/run_android_study.py --config configs/android/ui_voyager_qwen4b_variants.json report
```

`all`表示新配置声明的四组；`both`保留旧含义，只运行UI-Voyager和固定SPECTRUM。
也可把`all`换成某一个新arm，逐个执行；同一output root仍禁止多个并发拥有者。
资格探测按arm保存，但只是有界的真实环境接口检查，不证明多轮效果或最长上下文显存可容纳。
若探测没有足够真实成功/反馈，会保留signal_shortage并停止；不得制造标签绕过。

若尚未准备源代码/模型/设备/manifest，在新配置下依次使用原指南已有的
`fetch --source-only`、`setup-env`、`fetch`、`prepare`入口；这不会改变数据划分或模型pin。
每组仍保留完整合并checkpoint，新增两组会增加磁盘、采集和训练成本，旧双组容量估计不适用于四组。
更改SDK路径、GPU、输出根或预算须在正式seal前完成；保存实际执行SHA和资源回执。

## 验收与结果返回

原生AndroidWorld成功率仍是主指标；报告四组各轮实际成功率和缺失状态，
并保留每轮effective_tau、H/H_1、算子哈希、反馈来源与所有采集/训练成本。
固定版本与新版本缺失的结果不得填0，旧CPU测试不计作新实验结果。
每组rollout尝试数和可用优化器更新次数相同，不表示总tokens、动作数或FLOPs相同；
新增谱熵计算和参考采集也需要计入成本。

返回冻结study/config/manifest身份、各arm的 `round.json`、geometry的
`calibration.json` 与 `complete.json`、原生evaluation/训练回执、`report.json`和日志；
保留原始轨迹、截图及checkpoint供核验，但不要公开凭据、私有参数或大模型权重。
