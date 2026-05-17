"""自动瞄准主程序 - 优化版（解决圆周运动问题）"""
import os
import sys
import time
import ctypes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np
from inference.screen_capture import create_capture, get_center_region
from inference.detector import HeadDetector

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

def is_key_pressed(vk_code):
    return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0

VK_X1 = 0x05
VK_X2 = 0x06
VK_F12 = 0x7B
VK_F11 = 0x7A

# 加载配置
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

screen_w = config["screen_width"]
screen_h = config["screen_height"]
region = get_center_region(screen_w, screen_h, config["capture_size"])
region_x, region_y = region[0], region[1]
screen_cx, screen_cy = screen_w // 2, screen_h // 2

print("正在初始化...")
time.sleep(1)
capture = create_capture(region=region)

model_path = config["model_path"]
if not os.path.isabs(model_path):
    model_path = os.path.join(PROJECT_ROOT, model_path)
detector = HeadDetector(model_path, conf=config["confidence"], imgsz=config["imgsz"], half=config.get("half", False))

sensitivity = config.get("sensitivity", 0.8)
max_move = config.get("max_move", 150)


class KalmanFilter2D:
    """优化后的卡尔曼滤波器"""
    def __init__(self, process_noise=0.3, measurement_noise=0.3):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1000
        self.F = np.eye(4)
        self.F[0, 2] = 1
        self.F[1, 3] = 1
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x[0] = measurement[0]
            self.x[1] = measurement[1]
            self.initialized = True
            return measurement
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        z = np.array(measurement)
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return self.x[0], self.x[1]

    def reset_velocity(self):
        """重置速度分量，消除惯性"""
        self.x[2] = 0
        self.x[3] = 0

    def reset(self):
        self.initialized = False
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1000


# 目标追踪
current_target = None
target_lost_frames = 0
kalman = KalmanFilter2D(process_noise=0.3, measurement_noise=0.3)

# 稳定性检测
stable_frames = 0
STABLE_THRESHOLD = 2

# 速度阻尼：连续小偏移计数
small_offset_count = 0
SMALL_OFFSET_THRESHOLD = 10  # 连续 N 帧小偏移后重置速度

# 自动开火状态
auto_fire = False
prev_f11 = False

def click_mouse():
    """发送鼠标点击（减少阻塞时间）"""
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.005)  # 从 20ms 减少到 5ms
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def is_in_box(dx, dy, box_w, box_h):
    """检查准星偏移是否在框内"""
    return abs(dx) < box_w / 2 and abs(dy) < box_h / 2

def is_in_center(dx, dy, box_w, box_h):
    """检查准星偏移是否在框内（80%区域）"""
    return abs(dx) < box_w * 0.4 and abs(dy) < box_h * 0.4

def get_sensitivity(dx, dy):
    """分段灵敏度：大偏移快接近，小偏移精微调"""
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > 50:
        return 0.8   # 大偏移：快速接近
    elif dist > 10:
        return 0.6   # 中偏移：平稳追踪
    else:
        return 0.3   # 小偏移：精确微调

def select_target(heads):
    global current_target, target_lost_frames, kalman
    global stable_frames, small_offset_count

    if not heads:
        target_lost_frames += 1
        if target_lost_frames > 1:  # 丢失 1 帧就切换（更快）
            current_target = None
            kalman.reset()
            stable_frames = 0
            small_offset_count = 0
        return None, None, None, None

    target_lost_frames = 0

    # 转换为屏幕坐标
    targets = []
    for hx, hy, conf, bw, bh in heads:
        screen_x = hx + region_x
        screen_y = hy + region_y
        dx = screen_x - screen_cx
        dy = screen_y - screen_cy
        dist = (dx * dx + dy * dy) ** 0.5
        targets.append((dx, dy, conf, dist, screen_x, screen_y, bw, bh))

    # 追踪当前目标
    if current_target is not None:
        cur_x, cur_y = current_target
        best_same = None
        best_same_dist = float("inf")
        for t in targets:
            track_dist = ((t[4] - cur_x) ** 2 + (t[5] - cur_y) ** 2) ** 0.5
            if track_dist < best_same_dist:
                best_same_dist = track_dist
                best_same = t
        if best_same and best_same_dist < 300:
            current_target = (best_same[4], best_same[5])
            raw_dx, raw_dy = best_same[0], best_same[1]
            box_w, box_h = best_same[6], best_same[7]
        else:
            targets.sort(key=lambda t: t[3])
            best = targets[0]
            current_target = (best[4], best[5])
            raw_dx, raw_dy = best[0], best[1]
            box_w, box_h = best[6], best[7]
    else:
        targets.sort(key=lambda t: t[3])
        best = targets[0]
        current_target = (best[4], best[5])
        raw_dx, raw_dy = best[0], best[1]
        box_w, box_h = best[6], best[7]

    # 稳定性检测
    stable_frames += 1
    if stable_frames < STABLE_THRESHOLD:
        return None, None, None, None

    # 卡尔曼滤波（不再使用 EMA）
    kalman_dx, kalman_dy = kalman.update([raw_dx, raw_dy])

    # 速度阻尼：连续小偏移时重置速度
    if abs(raw_dx) < 10 and abs(raw_dy) < 10:
        small_offset_count += 1
        if small_offset_count >= SMALL_OFFSET_THRESHOLD:
            kalman.reset_velocity()
            small_offset_count = 0
    else:
        small_offset_count = 0

    return kalman_dx, kalman_dy, box_w, box_h


print("=" * 50)
print("自动瞄准系统已启动")
print("侧键后退 (X1) - 按住激活瞄准")
print("侧键前进 (X2) - 切换瞄准开关")
print("F11 - 切换自动开火")
print("F12 - 退出")
print("=" * 50)

toggle_active = False
prev_x2 = False
frame_count = 0

try:
    while True:
        if is_key_pressed(VK_F12):
            break

        x1_pressed = is_key_pressed(VK_X1)
        x2_pressed = is_key_pressed(VK_X2)
        if x2_pressed and not prev_x2:
            toggle_active = not toggle_active
            print(f"瞄准 {'ON' if toggle_active else 'OFF'}")
        prev_x2 = x2_pressed

        f11_pressed = is_key_pressed(VK_F11)
        if f11_pressed and not prev_f11:
            auto_fire = not auto_fire
            print(f"自动开火 {'ON' if auto_fire else 'OFF'}")
        prev_f11 = f11_pressed

        active = x1_pressed or toggle_active
        if not active:
            time.sleep(0.001)
            continue

        frame = capture.grab()
        if frame is None:
            continue

        frame_count += 1
        heads = detector.detect(frame)
        dx, dy, box_w, box_h = select_target(heads)

        if dx is not None and dy is not None:
            # 死区（增大到 5 像素）
            deadzone = 5
            if abs(dx) < deadzone and abs(dy) < deadzone:
                if auto_fire and box_w is not None and is_in_center(dx, dy, box_w, box_h):
                    click_mouse()
                continue

            # 钳制
            clamped_dx = max(-max_move, min(max_move, dx))
            clamped_dy = max(-max_move, min(max_move, dy))

            # 分段灵敏度
            sens = get_sensitivity(dx, dy)
            final_dx = int(clamped_dx * sens)
            final_dy = int(clamped_dy * sens)

            # 最小移动阈值（避免 1 像素的无效调用）
            if abs(final_dx) < 2 and abs(final_dy) < 2:
                if auto_fire and box_w is not None and is_in_center(dx, dy, box_w, box_h):
                    click_mouse()
                continue

            # 移动鼠标
            user32.mouse_event(MOUSEEVENTF_MOVE, final_dx, final_dy, 0, 0)

            # 自动开火（在框中心区域内才开火）
            if auto_fire and box_w is not None and is_in_center(dx, dy, box_w, box_h):
                click_mouse()

            if frame_count % 10 == 0:
                fire_status = " [FIRE]" if auto_fire and is_in_center(dx, dy, box_w, box_h) else ""
                print(f"[{frame_count}] 偏移({dx:.1f},{dy:.1f}) -> 移动({final_dx},{final_dy}) 灵敏度={sens:.1f}{fire_status}")

except KeyboardInterrupt:
    pass

capture.release()
print("\n已停止")
