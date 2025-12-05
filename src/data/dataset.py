import os
import random
import shutil
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, YOLO_DATA_DIR, SEED

class DatasetSplitter:
    def __init__(self, images_dir, labels_dir, output_base_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_base = Path(output_base_dir)
        
        self.train_images = self.output_base / "images/train"
        self.val_images = self.output_base / "images/val"
        self.train_labels = self.output_base / "labels/train"
        self.val_labels = self.output_base / "labels/val"
        
        random.seed(SEED)

    def split_dataset(self, split_ratio=0.8):
        # Create directories
        for path in [self.train_images, self.val_images, self.train_labels, self.val_labels]:
            path.mkdir(parents=True, exist_ok=True)

        # Get list of images
        image_files = [f for f in self.images_dir.glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        random.shuffle(image_files)

        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        print(f"Splitting dataset: {len(train_files)} train, {len(val_files)} val")

        self._move_files(train_files, self.train_images, self.train_labels)
        self._move_files(val_files, self.val_images, self.val_labels)

        print("Dataset dividido en train y val con estructura YOLO.")

    def _move_files(self, files, img_dest, label_dest):
        for img_file in files:
            label_file = self.labels_dir / (img_file.stem + ".txt")
            
            shutil.copy(img_file, img_dest / img_file.name)
            if label_file.exists():
                shutil.copy(label_file, label_dest / label_file.name)

if __name__ == "__main__":
    splitter = DatasetSplitter(PROCESSED_DATA_DIR / "images", PROCESSED_DATA_DIR / "labels", YOLO_DATA_DIR)
    splitter.split_dataset()
