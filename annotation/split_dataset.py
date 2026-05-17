"""数据集划分脚本 - 将标注数据按比例划分到 train/val/test"""
import os
import shutil
import random
import glob


def split_dataset(captures_dir, dataset_dir, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05, seed=42):
    """将截图和标注文件划分到数据集目录

    Args:
        captures_dir: 截图目录（包含图片和对应的 .txt 标注文件）
        dataset_dir: 数据集输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    random.seed(seed)

    # 收集所有有标注的图片
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(glob.glob(os.path.join(captures_dir, ext)))

    # 只保留有对应标注文件的图片
    valid_files = []
    for img_path in image_files:
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(label_path):
            valid_files.append(img_path)

    if not valid_files:
        print(f"错误: 在 {captures_dir} 中没有找到带标注的图片")
        print("请先用 LabelImg 标注图片，确保 .txt 文件和图片在同一目录")
        return

    random.shuffle(valid_files)

    total = len(valid_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": valid_files[:train_end],
        "val": valid_files[train_end:val_end],
        "test": valid_files[val_end:],
    }

    for split_name, files in splits.items():
        img_dir = os.path.join(dataset_dir, "images", split_name)
        lbl_dir = os.path.join(dataset_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_path in files:
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            label_path = os.path.splitext(img_path)[0] + ".txt"

            shutil.copy2(img_path, os.path.join(img_dir, filename))
            shutil.copy2(label_path, os.path.join(lbl_dir, name_no_ext + ".txt"))

        print(f"{split_name}: {len(files)} 张图片")

    print(f"\n总计: {total} 张图片已划分到 {dataset_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="划分数据集")
    parser.add_argument("--input", "-i", required=True, help="截图目录（包含图片和标注）")
    parser.add_argument("--output", "-o", default="dataset", help="数据集输出目录")
    parser.add_argument("--train", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test", type=float, default=0.05, help="测试集比例")
    args = parser.parse_args()

    split_dataset(args.input, args.output, args.train, args.val, args.test)
