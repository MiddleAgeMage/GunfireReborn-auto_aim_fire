"""鼠标控制器 - Windows SendInput API"""
import ctypes

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT)]


class SmoothMouseController:
    def __init__(self, sensitivity=1.0, max_move_pixels=200):
        self.sensitivity = sensitivity
        self.max_move = max_move_pixels
        self.user32 = ctypes.windll.user32
        # 预创建 INPUT 结构体，避免重复分配
        self._ii = INPUT()
        self._ii.type = INPUT_MOUSE
        self._ii.mi.dwFlags = MOUSEEVENTF_MOVE

    def move_relative(self, dx, dy):
        dx = max(-self.max_move, min(self.max_move, dx))
        dy = max(-self.max_move, min(self.max_move, dy))
        dx = int(dx * self.sensitivity)
        dy = int(dy * self.sensitivity)
        if dx == 0 and dy == 0:
            return
        self._ii.mi.dx = dx
        self._ii.mi.dy = dy
        self.user32.SendInput(1, ctypes.pointer(self._ii), ctypes.sizeof(self._ii))

    def move_smooth(self, target_dx, target_dy, steps=2):
        # 直接移动，不做分步（减少延迟）
        self.move_relative(target_dx, target_dy)
