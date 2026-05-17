"""性能基准测试 - 测量各组件延迟"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def benchmark_capture(capture, iterations=100):
    """测量屏幕截图延迟"""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        frame = capture.grab()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return {
        "mean_ms": np.mean(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }


def benchmark_inference(detector, frame, iterations=100):
    """测量 YOLO 推理延迟"""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        heads = detector.detect(frame)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return {
        "mean_ms": np.mean(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "detections": len(heads),
    }


def benchmark_mouse(controller, iterations=100):
    """测量鼠标移动延迟"""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        controller.move_relative(1, 0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        controller.move_relative(-1, 0)
    return {
        "mean_ms": np.mean(times),
        "p99_ms": np.percentile(times, 99),
    }


def run_benchmark():
    import yaml
    from inference.screen_capture import create_capture, get_center_region
    from inference.detector import HeadDetector
    from inference.mouse_controller import SmoothMouseController

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)

    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    region = get_center_region(config["screen_width"], config["screen_height"], config["capture_size"])

    print("=" * 50)
    print("性能基准测试")
    print("=" * 50)

    # 测试截图
    print("\n[1/3] 测试屏幕截图...")
    capture = create_capture(region=region)
    cap_stats = benchmark_capture(capture, iterations=50)
    print(f"  平均: {cap_stats['mean_ms']:.2f}ms | P95: {cap_stats['p95_ms']:.2f}ms | P99: {cap_stats['p99_ms']:.2f}ms")

    # 测试推理
    print("\n[2/3] 测试 YOLO 推理...")
    frame = capture.grab()
    if frame is not None:
        model_path = config["model_path"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
        if os.path.exists(model_path):
            detector = HeadDetector(model_path, conf=config["confidence"], imgsz=config["imgsz"], half=config.get("half", False))
            det_stats = benchmark_inference(detector, frame, iterations=30)
            print(f"  平均: {det_stats['mean_ms']:.2f}ms | P95: {det_stats['p95_ms']:.2f}ms | 检测数: {det_stats['detections']}")
        else:
            print(f"  模型未找到: {model_path}")
            det_stats = {"mean_ms": 0}
    else:
        print("  截图失败")
        det_stats = {"mean_ms": 0}

    # 测试鼠标
    print("\n[3/3] 测试鼠标移动...")
    mouse_ctrl = SmoothMouseController()
    mouse_stats = benchmark_mouse(mouse_ctrl, iterations=50)
    print(f"  平均: {mouse_stats['mean_ms']:.2f}ms | P99: {mouse_stats['p99_ms']:.2f}ms")

    # 总结
    total = cap_stats["mean_ms"] + det_stats["mean_ms"] + mouse_stats["mean_ms"]
    print("\n" + "=" * 50)
    print(f"总延迟: {total:.2f}ms | 预期 FPS: {1000/total:.1f}")
    print("=" * 50)

    capture.release()


if __name__ == "__main__":
    run_benchmark()
