# GUIAGENT：纯 Windows + NVIDIA 96GB 显存，双 GUI Benchmark

**实现状态：** 代码、PowerShell 安装、AndroidWorld 第二 Benchmark、训练入口已经写入仓库；我没有连接你的 Windows 主机，不能宣称已经在你的 NVIDIA 卡上跑通真实 AVD 或完成训练。先按下面的 Pilot 实测。

## 1. 两个 Benchmark 到底是什么？

| 项目 | MobileGym | AndroidWorld |
| --- | --- | --- |
| 模拟方式 | Playwright + Chromium 的网页手机 | 真正的 Android Emulator（Pixel 6、Android 13） |
| 环境反馈 | 程序化状态验证器 | task.is_successful(env) 原生验证器 |
| 输入模型 | 截图 + 历史动作 | 截图 + 历史动作 |
| 模型 | 同一个 Qwen3-VL-4B BF16 | 同一个 Qwen3-VL-4B BF16 |
| 方法 | Feedback SD / Fixed / Learned SPECTRUM | 同上 |
| WSL / Linux | 不需要 | 不需要 |
| 虚拟化 | 不需要 | 必须有 Windows WHPX / CPU 虚拟化 |

96GB 指 GPU 显存 VRAM，并不是 Windows 系统 RAM。建议物理内存至少 64GB；完整多臂多轮结果建议预留至少 500GB 可用磁盘，最好 1TB。首次运行依然需要 GPU Peak Memory 实测。

**特别重要：** 如果这台 Windows 电脑其实是云端 Windows 虚拟机，提供商可能关闭了 nested virtualization。96GB NVIDIA GPU 无法解决这个问题；AndroidWorld 会被 WHPX 检查阻止，MobileGym 不受影响。不要用 WSL 或软件慢速模拟来伪装原生 Windows 已通过测试。

## 2. 一次性安装

在 Windows PowerShell 中进入 GUIAGENT 仓库，先安装 Windows 系统软件：

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\setup_windows.ps1 -Benchmark Both -InstallSystemTools

安装后关闭 PowerShell 并重新打开。然后在 Android Studio 的 SDK Manager 里勾选 Android SDK Command-line Tools (latest)。在 Windows 功能中启用 Windows Hypervisor Platform (WHPX)，如果提示则重启 Windows。

再运行完整安装（官方 PyTorch 2.8 CUDA 12.8、Qwen3-VL-4B 模型、两个环境及其依赖、AVD、Chromium、MobileGym 1.9GB 数据包）：

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\setup_windows.ps1 -Benchmark Both -SetupAndroidSDK -DownloadModel -DownloadMobileGymData

安装程序不会自动升级 NVIDIA 驱动，也不会启动模型训练。若你已经安装这些系统组件，可直接运行第二条命令，不要重复安装系统软件。

验证本机依赖与 GPU：

    $py = ".\.venv-guiagent\Scripts\python.exe"
    & $py scripts\windows\static_check.py
    & $py scripts\windows\preflight_guiagent.py --benchmark both
    & $py -m pip check
    & $py -m pytest tests\test_android_windows_native.py tests\test_androidworld_feedback.py tests\test_windows_gui_contract.py tests\test_mobilegym_feedback.py -q

检查输出应有真实的 GPU 型号、CUDA Runtime、显存 GiB、BF16 支持、Android AVD、WHPX 检查结果。即使静态测试通过，也还不能算 AndroidWorld 真实运行成功。

如果 PowerShell 不允许直接执行 .ps1，可以在当前窗口执行：

    Set-ExecutionPolicy -Scope Process Bypass

这不会永久修改全局执行策略；如组织策略禁止，应由系统管理员决定。

## 3. AndroidWorld（新增第二 Benchmark）

**打开一个 PowerShell 窗口启动 Android 模拟器：**

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start_androidworld.ps1

启动参数为 Pixel 6 / API33 / AndroidWorldAvd，控制台端口 5554，gRPC 8554；无图形桌面可额外加 -Headless，但依然需要 WHPX。

**在另一个 PowerShell 窗口执行：**

    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage setup-env
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage env-check
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage prepare
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage plan

先跑 4 个训练模板、8 个测试任务、1 轮小实验：

    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Stage run

只有 Pilot 确认真实截图、动作执行、环境评分、几何和训练都正确，才正式运行：

    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage prepare
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage plan
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage run
    .\scripts\windows\run_benchmark.ps1 -Benchmark AndroidWorld -Full -Stage report

正式配置包含 16 种训练任务模板、每模板 2 个实例、116 道测试、3 轮迭代。Round 0 是 128 次原生探索，不进行 SFT；然后分别训练 Feedback SD、固定谱 SPECTRUM、最新连续 Learned Gains SPECTRUM。三组使用同样的 Qwen3-VL 起点和 rollout 尝试预算；AndroidWorld 的 task.is_successful 真实评分才是 Verified Reward。

注意：这里和历史 UI-Voyager 的 GRSD+RFT 实验不同，不能直接把历史分数当作同预算对照；新实验不使用独立 Calibration 答案，也不把测试轨迹放回训练。

## 4. MobileGym（原有主线，保留）

在另一个 PowerShell 窗口启动网页模拟器：

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start_mobilegym.ps1

然后运行：

    .\scripts\windows\run_benchmark.ps1 -Benchmark MobileGym -Stage plan
    .\scripts\windows\run_benchmark.ps1 -Benchmark MobileGym -Stage env-check
    .\scripts\windows\run_benchmark.ps1 -Benchmark MobileGym -Stage run

完整 32 训练模板 / 256 测试任务 / 3 轮实验需要显式添加 -Full。MobileGym 通过浏览器模拟，并不需要 Android Studio 或 WHPX。它的旧 Pilot/结果目录不会被新的 AndroidWorld 入口覆盖。

## 5. 出结果后先检查什么？

主要看每轮真实 Success Rate、native/spectral 成功轨迹发现数、总交互次数、学生反馈 KL、Learnt Gain 是否回退恒等算子，以及 BF16 GPU Peak VRAM。

不能把 MobileGym 256 题 SR 与 AndroidWorld 116 题 SR 求一个平均值；两套环境独立报告。96GB 解决的是较大的多模态训练显存预算，**不代表模拟器、Python 兼容或 Long-Horizon 成功率一定通过**。

原英文详细说明：[Windows 原生双环境手册](WINDOWS_NATIVE_96GB_GUI_BENCHMARKS.md)。
