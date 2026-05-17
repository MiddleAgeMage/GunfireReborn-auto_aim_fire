"""游戏截图采集工具 - 鼠标侧键截图用于训练数据收集"""
import os
import time
import threading
from datetime import datetime

import mss
import numpy as np
import cv2
from pynput import keyboard, mouse


class CaptureTool:
    def __init__(self, output_dir="captures", screen_width=2560, screen_height=1440):
        self.output_dir = output_dir
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.count = 0
        self.session_dir = None
        self.running = True
        self.auto_capture = False
        self.auto_interval = 0.5

    def create_session(self):
        """创建新的截图会话目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.count = 0
        print(f"截图保存目录: {self.session_dir}")

    def capture(self):
        """截取全屏并保存"""
        if self.session_dir is None:
            self.create_session()

        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
        frame = np.array(screenshot)[:, :, :3]

        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.png"
        filepath = os.path.join(self.session_dir, filename)
        cv2.imwrite(filepath, frame)

        self.count += 1
        print(f"[{self.count}] 截图已保存: {filename}")
        return filepath

    def on_click(self, x, y, button, pressed):
        """鼠标按键回调"""
        try:
            if button == mouse.Button.x1 and pressed:
                # 侧键后退 = 单次截图
                self.capture()
            elif button == mouse.Button.x2 and pressed:
                # 侧键前进 = 切换连续截图
                self.auto_capture = not self.auto_capture
                if self.auto_capture:
                    print(f"连续截图模式开启 (间隔 {self.auto_interval}s)")
                    threading.Thread(target=self._auto_capture_loop, daemon=True).start()
                else:
                    print("连续截图模式关闭")
        except Exception as e:
            print(f"截图失败: {e}")

    def on_press(self, key):
        """键盘按下回调（用于退出）"""
        if key == keyboard.Key.f12:
            self.running = False
            return False

    def _auto_capture_loop(self):
        """连续截图循环"""
        while self.auto_capture and self.running:
            self.capture()
            time.sleep(self.auto_interval)

    def run(self):
        """启动截图工具"""
        self.create_session()
        print("=" * 50)
        print("游戏截图采集工具")
        print("=" * 50)
        print("侧键后退 (X1) - 单次截图")
        print("侧键前进 (X2) - 开始/停止连续截图")
        print("F12           - 退出程序")
        print("=" * 50)

        mouse_listener = mouse.Listener(on_click=self.on_click)
        keyboard_listener = keyboard.Listener(on_press=self.on_press)
        mouse_listener.start()
        keyboard_listener.start()
        keyboard_listener.join()
        mouse_listener.stop()

        print(f"\n本次共截图 {self.count} 张，保存在: {self.session_dir}")


if __name__ == "__main__":
    tool = CaptureTool()
    tool.run()
