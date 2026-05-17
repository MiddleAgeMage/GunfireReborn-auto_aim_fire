"""屏幕捕获模块"""
import numpy as np


def get_center_region(screen_w=2560, screen_h=1440, region_size=1280):
    """计算以屏幕中心为基准的截图区域"""
    cx, cy = screen_w // 2, screen_h // 2
    half = region_size // 2
    return (cx - half, cy - half, cx + half, cy + half)


class DXCamCapture:
    """使用 dxcam 的屏幕捕获"""

    def __init__(self, region=None):
        import dxcam
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.region = region

    def grab(self):
        # 每次单独截图，不用连续捕获模式
        frame = self.camera.grab(region=self.region)
        return frame

    def release(self):
        self.camera.release()


class MSSCapture:
    """使用 mss 的屏幕捕获"""

    def __init__(self, region=None):
        import mss
        self._mss_mod = mss
        if region:
            self.monitor = {
                "left": region[0], "top": region[1],
                "width": region[2] - region[0], "height": region[3] - region[1],
            }
        else:
            self.monitor = None

    def grab(self):
        with self._mss_mod.mss() as sct:
            monitor = self.monitor if self.monitor else sct.monitors[1]
            screenshot = sct.grab(monitor)
        frame = np.array(screenshot)[:, :, :3]
        return frame

    def release(self):
        pass


def create_capture(region=None, prefer_dxcam=True):
    """创建屏幕捕获实例"""
    if prefer_dxcam:
        try:
            return DXCamCapture(region=region)
        except Exception as e:
            print(f"dxcam 失败 ({e})，用 mss")
    return MSSCapture(region=region)
