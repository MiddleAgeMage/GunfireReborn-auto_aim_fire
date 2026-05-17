"""自动瞄准 - 直接运行此文件"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 运行主程序
from inference.auto_aim import *
