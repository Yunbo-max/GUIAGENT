# MobileGym 反馈训练与评测运行复盘（2026-11-14）

## 结论摘要

- 本轮 MobileGym 反馈训练已完成；正式测试评测在用户要求停止时仅完成 **95/256** 条。
- 已完成部分有 **2 条成功、93 条失败，成功率 2.1%**。这只是部分结果，不能当作最终成绩。
- 运行持续约 **18 小时 15 分钟**，其中正式评测阶段约 **15 小时 16 分钟**。按评测阶段平均吞吐粗估，剩余 161 条还需约 **26 小时**；此估算不保证后续速度相同。
- 训练峰值显存不高（Reserved 约 **20.0 GiB**），但整体评测吞吐慢，且当前部分成功率低。
- 没有在同一 256 个测试任务上完成初始模型基线，因此**不能判断反馈训练是否提升成功率**。
- 进程是按用户要求停止，不是运行时崩溃。已保留已完成轨迹、截图、模型检查点和日志；另有 4 条未完成轨迹仅保留了部分截图。

## 运行设置与训练结果

- 基准：MobileGym；方案：`feedback_sd`，round 1。
- 模型：Qwen3-VL-4B BF16；指定物理 GPU 1；PyTorch allocator 上限 40 GiB。
- 训练设置：microbatch 8、gradient accumulation 8、有效 batch 64。
- 反馈训练：5,167 个训练样例、64 次 optimizer update、平均训练 loss 0.0904。
- 训练记录的峰值 CUDA allocated 为 **12.81 GiB**，peak reserved 为 **19.98 GiB**。因此训练没有接近 40 GiB allocator 上限。
- 日志报告 2,048 条 native 训练轨迹、70 条成功；反馈训练 rollout 另报告 2,048 条、79 条成功。这些是训练侧数据，不能替代独立测试集基线。

## 正式评测快照

评测于 `2026-11-13 20:24:39 UTC` 开始。最后一条运行日志为 `2026-11-14 11:40:30 UTC`，记录 3,075 次 generation call，阶段仍为 evaluation。

停止时：

- 评测任务总数：256。
- 已写入完整 `episode.json`：95 条，覆盖 95 个不同 task。
- 成功 2、失败 93；部分成功率 **2/95 = 2.1%**。
- 已完成轨迹累计 1,888 个 action step，平均每条约 19.9 步。
- 有 4 个未完成 task 目录，没有最终 `episode.json`；它们不计入 95 条完成记录。
- 评测目录没有最终 aggregate `result.json`。
- 两个成功任务为 `reddit.Reddit_UpdateProfileBio` 和 `reddit.Reddit_AdvancedPrivacyToggles`。

部分失败诊断中，最常见的 verifier issue 是 `answer_sheet.submitted` 未提交（10 条）。其余失败分散在不同应用状态字段，当前样本量不足以认定单一主因。

## 速度与资源观察

- 评测阶段约 15 小时 16 分钟完成 95 条，观察到的平均吞吐约 **6.2 条/小时**。按这个阶段平均值外推，尚余 161 条约需 25.9 小时；任务长度和生成耗时波动很大，实际时间可能不同。
- 日志中约 3,076 条 `generation_completed` 事件：单次生成耗时平均约 19 秒、中位数 17.9 秒、P90 36.7 秒，最大 313.6 秒。生成等待之外还包含浏览器动作、环境状态验证和任务收尾时间。
- 64 次训练更新从 `18:32:05 UTC` 到 `20:24:30 UTC`，墙钟时间约 1 小时 52 分钟（含训练与检查点写入）。
- 停止前的系统级 `nvidia-smi` 快照：GPU 1 总显存 97,887 MiB，已用 69,702 MiB、空闲 27,737 MiB，利用率 92%；GPU 0 同时也有约 91% 利用率。该数据是**整张卡的系统级快照**，不能全部归因到本实验进程；训练自身记录的 Reserved 峰值约 20 GiB。
- 运行期间同时有 4 个 MobileGym 环境任务目录处于未完成状态，但本次没有记录 GPU kernel、模型队列等待、浏览器等待和 verifier 用时的分项 profiler。因此能够确认“端到端吞吐偏慢”，但还不能把瓶颈归结为单一环节。下一轮优化前应分别记录这些等待时间，再决定是否增加环境并发、改推理 batching 或调整截图/上下文成本。

## 是否证明反馈训练有效？

目前不能得出正向结论。部分反馈模型测试成功率为 2.1%，绝对表现偏低；同时 95/256 未覆盖全部测试任务。`round0_native` 的 70/2,048 和反馈 rollout 的 79/2,048 均为训练侧数据，不是同一 held-out 测试集上的初始模型评测，不能直接与 2/95 比较。

要回答“是否有提升”，需要在同一版本的 256 个测试任务、同一 verifier 和同一运行协议下，补跑初始模型基线，并完成反馈模型剩余评测。基线评测本次没有启动。

## 停止与数据保留

用户要求停止所有任务后，已终止本轮 `run.py mobilegym` 进程树及其 MobileGym Vite 服务；核验时 Python 运行进程已不存在，`127.0.0.1:4173` 已停止响应。停止不是 crash，日志中没有本次运行的 `failed` 事件；日志在 evaluation 中途结束。

以下本地实验产物未删除，未加入 Git：

- `runs/mobilegym_resume_20261108/mobilegym/full/round0_native/`
- `runs/mobilegym_resume_20261108/mobilegym/full/feedback_sd/round_01/native/`
- `runs/mobilegym_resume_20261108/mobilegym/full/feedback_sd/round_01/checkpoint/`
- `runs/mobilegym_resume_20261108/mobilegym/full/feedback_sd/round_01/eval/`
- `runs/mobilegym_resume6_gpu1_20261113/stdout.log` 与 `stderr.log`

AndroidWorld 本轮未运行，仍保持暂停。
