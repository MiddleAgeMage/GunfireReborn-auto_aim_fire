# 枪火重生-基于YOLOv8的自动瞄准系统实践

## 项目概述

基于 YOLOv8 + TensorRT 的实时头部检测与自动瞄准系统，用于 枪火重生 离线游戏。

**技术栈**: Python 3.8 + PyTorch 2.2.2 + CUDA 12.1 + TensorRT 10.16 + RTX 4060 Laptop

**核心功能**:
- 实时屏幕截图 + YOLOv8 头部检测
- 卡尔曼滤波平滑追踪
- 自动瞄准 + 自动开火
- 侧键控制开关

---

## 开发历程

### 第一阶段：环境搭建与基础功能

**目标**: 建立项目结构，实现基本的截图 → 检测 → 鼠标移动流程

**完成内容**:
- 创建项目目录结构 (`auto_aim/`)
- 安装依赖：mss, dxcam, pynput, ultralytics, pywin32
- 实现截图采集工具 (`capture_tool.py`)
- 实现 YOLOv8 训练脚本 (`train.py`)
- 实现基础检测器 (`detector.py`)
- 实现鼠标控制器 (`mouse_controller.py`)

**遇到的问题**:
- `mss` 跨线程调用报错 `_thread._local` object has no attribute `srcdc`
  - **解决**: 每次截图时创建新的 mss 实例
- `mss` 截图颜色通道 BGR/RGB 顺序错误，蓝色变红色
  - **解决**: 删除错误的 `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` 转换

---

### 第二阶段：数据采集与模型训练

**目标**: 收集游戏截图，标注头部，训练 YOLOv8 模型

**完成内容**:
- 使用 `labelImg` 标注 357 张截图（单类 `head`）
- 划分数据集：train 285 / val 53 / test 19
- 使用 `yolov8n.pt` 预训练模型微调
- 训练参数：epochs=150, imgsz=640, batch=16

---

### 第三阶段：推理优化与启动方式

**目标**: 提高推理速度，解决启动方式问题

**完成内容**:
- 安装 `onnxruntime-gpu`，尝试 CUDA 加速（失败，cuDNN 版本不匹配）
- 改用 `.pt` 模型直接推理（PyTorch CUDA），性能更好
- 安装 TensorRT，导出 `.engine` 模型，推理速度提升 3 倍

**性能对比**:

| 方案 | 推理时间 | 总 FPS |
|------|---------|--------|
| PyTorch + CPU | 43ms | 22 |
| PyTorch + CUDA | 20ms | 35 |
| TensorRT + CUDA | 6.6ms | 61 |

**遇到的问题**:
- `.bat` 文件启动后鼠标不移动（输入上下文绑定在控制台窗口）
  - **解决**: 改用命令行直接运行 `python aim.py`
- `dxcam` 全屏模式下崩溃（Desktop Duplication API 不支持独占全屏）
  - **解决**: 使用无边框全屏模式

---

### 第四阶段：鼠标移动与游戏兼容性

**目标**: 解决鼠标在游戏内不移动的问题

**问题**: `SendInput` 和 `pydirectinput` 在游戏内无效（仅在桌面有效）

**排查过程**:
1. 测试 `mouse_event` → 桌面有效，游戏内无效
2. 测试 `pydirectinput` → 桌面有效，游戏内无效
3. 测试 `SendInput` → 桌面有效，游戏内无效
4. 发现：单独运行移动脚本在游戏内有效，但 `auto_aim.py` 中无效

**根因**: `pynput` 监听器干扰了鼠标输出上下文

**解决**: 去掉 `pynput`，改用 `GetAsyncKeyState` 检测按键

---

### 第五阶段：平滑追踪与防抖

**目标**: 解决准星抖动和圆周运动问题

**问题分析**:
准星绕目标圆周运动的原因：
1. 卡尔曼滤波器恒速模型惯性过大（process_noise=0.03 太低）
2. 卡尔曼 + EMA 双重滤波导致相位滞后
3. 死区过小（2像素）+ 量化误差导致无法收敛
4. `click_mouse()` 的 sleep 阻塞主循环

**解决方案**:

| 问题 | 修改 |
|------|------|
| 双重滤波相位滞后 | 删除 EMA，仅用卡尔曼滤波 |
| 卡尔曼惯性过大 | process_noise 从 0.03 提高到 0.3 |
| 死区过小 | 从 2 像素增大到 5 像素 |
| 灵敏度一刀切 | 分段灵敏度：大偏移 0.8，中偏移 0.6，小偏移 0.3 |
| click_mouse 阻塞 | sleep 从 20ms 减少到 5ms |
| 开火范围太大 | 改为框中心 60% 区域才开火 |
| 速度残余 | 连续 10 帧小偏移后重置卡尔曼速度 |
| 无效移动 | 最小移动阈值 2 像素 |

---

### 第六阶段：快速切换与开火优化

**目标**: 提高目标切换速度，减少开火延迟

**完成内容**:
- 目标切换：丢失 1 帧立即切换（之前 3 帧）
- 开火区域：框内 80% 区域开火（之前 60%）
- 置信度：降低到 0.25（之前 0.45）

---

## 当前系统架构

```
auto_aim/
├── aim.py                    # 启动入口
├── config.yaml               # 配置文件
├── inference/
│   ├── auto_aim.py           # 主程序（卡尔曼滤波 + 自动开火）
│   ├── screen_capture.py     # 屏幕捕获（dxcam/mss）
│   ├── detector.py           # YOLOv8 检测器
│   └── mouse_controller.py   # 鼠标控制
├── training/
│   ├── train.py              # 训练脚本
│   └── runs/                 # 训练结果
├── data_collection/
│   └── capture_tool.py       # 截图采集工具
└── dataset/                  # 训练数据集
```

---

## 关键配置参数

```yaml
# config.yaml
screen_width: 2560
screen_height: 1440
capture_size: 1280          # 截图区域大小
confidence: 0.25            # 检测置信度阈值
imgsz: 480                  # YOLO 输入分辨率
sensitivity: 0.8            # 鼠标灵敏度
max_move: 150               # 单帧最大移动像素
```

---

## 使用方法

**启动**:
```bash
命令行指定python解释器运行auto_aim/aim.py
如：C:/Users/xxxx/.conda/envs/py38/python.exe C:/xxx/auto_aim/aim.py
```

**控制**:
- `X1`（侧键后退）— 按住临时激活瞄准
- `X2`（侧键前进）— 切换瞄准开关
- `F11` — 切换自动开火
- `F12` — 退出

---

## 未来优化方向

1. **GPU 直接截图**: 使用 DXGI + CUDA 互操作，消除 CPU-GPU 内存拷贝
2. **ByteTrack 目标追踪**: 为同一目标分配 ID，结合卡尔曼滤波预测修正
3. **模型重训练**: 针对小目标优化数据增强（scale=0.5）
4. **TensorRT INT8 量化**: 进一步加速推理
5. **异步开火**: 开火逻辑移到单独线程，避免阻塞主循环

---

## 依赖列表

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
mss>=9.0.0
dxcam>=0.4.0
tensorrt>=10.0.0
pyyaml>=6.0
```

---

## 环境信息

- **操作系统**: Windows 11 Pro
- **GPU**: NVIDIA GeForce RTX 4060 Laptop
- **CUDA**: 12.1
- **Python**: 3.8.19 (Anaconda py38 环境)
- **PyTorch**: 2.2.2
- **TensorRT**: 10.16.1.11
