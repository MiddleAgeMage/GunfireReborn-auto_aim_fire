"""YOLOv8 头部检测训练脚本"""
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


def train():
    model = YOLO("yolov8n.pt")

    # 数据集配置使用绝对路径
    data_yaml = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")

    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        patience=30,
        save=True,
        save_period=10,
        project="runs/train",
        name="head_detect",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        close_mosaic=15,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        scale=0.3,
        translate=0.1,
    )

    print(f"\n训练完成！最佳模型保存在: runs/train/head_detect/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
