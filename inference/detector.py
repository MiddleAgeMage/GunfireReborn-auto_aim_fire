"""YOLOv8 头部检测器封装"""
import numpy as np


class HeadDetector:
    def __init__(self, model_path, conf=0.45, imgsz=480, device="0", half=True):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.half = half
        # 预热推理
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, conf=self.conf, imgsz=self.imgsz, verbose=False, device=self.device, half=self.half)
        print(f"模型加载完成: {model_path} (conf={conf}, imgsz={imgsz}, FP16={half})")

    def detect(self, frame):
        """检测头部位置

        Args:
            frame: numpy array (H, W, 3) BGR

        Returns:
            list of (x_center, y_center, confidence, width, height) 像素坐标
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
            half=self.half,
        )

        heads = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xywh.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    heads.append((float(box[0]), float(box[1]), float(conf), float(box[2]), float(box[3])))

        return heads
