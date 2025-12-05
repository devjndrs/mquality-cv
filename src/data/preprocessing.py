import os
import cv2
import numpy as np
import albumentations as A
import random
import shutil
from collections import Counter
from pathlib import Path
from src.config import TEMP_DATA_DIR, PROCESSED_DATA_DIR, SEED

class Preprocessor:
    def __init__(self, raw_dir, annotations_dir):
        self.raw_dir = Path(raw_dir)
        self.annotations_dir = Path(annotations_dir)
        self.resize_dir = TEMP_DATA_DIR / "resize"
        self.normalized_dir = TEMP_DATA_DIR / "normalized"
        self.augmented_img_dir = TEMP_DATA_DIR / "augmented" / "images"
        self.augmented_label_dir = TEMP_DATA_DIR / "augmented" / "labels"
        self.balanced_img_dir = TEMP_DATA_DIR / "balanced" / "images"
        self.balanced_label_dir = TEMP_DATA_DIR / "balanced" / "labels"
        
        # Create directories
        for d in [self.resize_dir, self.normalized_dir, self.augmented_img_dir, 
                  self.augmented_label_dir, self.balanced_img_dir, self.balanced_label_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.set_seed(SEED)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        print(f"Semillas fijadas para reproducibilidad (SEED={seed})")

    def resize_images(self, target_size=(640, 640)):
        print("Iniciando redimensionamiento...")
        for img_path in self.raw_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    resized = cv2.resize(img, target_size)
                    cv2.imwrite(str(self.resize_dir / img_path.name), resized)
        print("Redimensionamiento completado.")

    def normalize_images(self):
        print("Iniciando normalización...")
        for img_path in self.resize_dir.glob("*"):
            img = cv2.imread(str(img_path))
            if img is not None:
                normalized = img.astype(np.float32) / 255.0
                # Saving as uint8 for compatibility, though normalization usually implies float for training.
                # If the intention is just to have them ready, saving as image might lose precision if not careful.
                # The notebook saved them as (normalized * 255).astype(np.uint8), which effectively reverts normalization for storage.
                # We will follow the notebook's logic but keep in mind this step might be redundant if loading transforms do it.
                cv2.imwrite(str(self.normalized_dir / img_path.name), (normalized * 255).astype(np.uint8))
        print("Normalización completada.")

    def augment_data(self):
        print("Iniciando data augmentation...")
        transform = A.Compose([
            A.Rotate(limit=20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Blur(blur_limit=3, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        for img_path in self.normalized_dir.glob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            label_path = self.annotations_dir / (img_path.stem + ".txt")
            
            if not label_path.exists():
                print(f"⚠ No label para {img_path.name}, se omite.")
                continue

            image = cv2.imread(str(img_path))
            bboxes, labels = self.read_yolo_labels(label_path)

            if len(bboxes) == 0:
                # Save original if no boxes
                cv2.imwrite(str(self.augmented_img_dir / img_path.name), image)
                self.save_yolo_labels(self.augmented_label_dir / label_path.name, [], [])
                continue

            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                
                # Save augmented image
                aug_img_name = f"aug_{img_path.name}"
                cv2.imwrite(str(self.augmented_img_dir / aug_img_name), transformed['image'])
                
                # Save augmented labels
                aug_label_name = f"aug_{label_path.name}"
                self.save_yolo_labels(self.augmented_label_dir / aug_label_name, 
                                    transformed['bboxes'], transformed['class_labels'])
                
                # Also save original to augmented dir for next steps
                cv2.imwrite(str(self.augmented_img_dir / img_path.name), image)
                self.save_yolo_labels(self.augmented_label_dir / label_path.name, bboxes, labels)
                
            except Exception as e:
                print(f"Error augmenting {img_path.name}: {e}")

        print("Data augmentation completado.")

    def balance_classes(self):
        print("Iniciando balanceo de clases...")
        class_counts = Counter()
        label_files = list(self.augmented_label_dir.glob("*.txt"))

        for lf in label_files:
            _, labels = self.read_yolo_labels(lf)
            class_counts.update(labels)

        print("📊 Distribución inicial de clases:", class_counts)
        if not class_counts:
            print("No classes found.")
            return

        max_count = max(class_counts.values())

        # Copy all current files to balanced dir
        for img_path in self.augmented_img_dir.glob("*"):
            shutil.copy(img_path, self.balanced_img_dir)
        for label_path in self.augmented_label_dir.glob("*"):
            shutil.copy(label_path, self.balanced_label_dir)

        # Oversampling
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        for cls_id, count in class_counts.items():
            if count < max_count:
                needed = max_count - count
                print(f"🔄 Balanceando clase {cls_id}: +{needed} instancias")
                
                imgs_with_cls = []
                for lf in label_files:
                    bboxes, labels = self.read_yolo_labels(lf)
                    if cls_id in labels:
                        imgs_with_cls.append(lf)

                for i in range(needed):
                    if not imgs_with_cls:
                        break
                    
                    # Pick random image with that class
                    label_file = random.choice(imgs_with_cls)
                    img_file = self.augmented_img_dir / (label_file.stem.replace('aug_', '').replace('aug_', '') + ".jpg") # Try to find original or aug
                    if not img_file.exists():
                         # Try matching extension
                         for ext in ['.jpg', '.png', '.jpeg']:
                             img_file = self.augmented_img_dir / (label_file.stem + ext)
                             if img_file.exists():
                                 break
                    
                    if not img_file.exists():
                        continue

                    image = cv2.imread(str(img_file))
                    bboxes, labels = self.read_yolo_labels(label_file)

                    try:
                        transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                        
                        aug_img_name = f"{img_file.stem}_bal{i}.jpg"
                        cv2.imwrite(str(self.balanced_img_dir / aug_img_name), transformed['image'])
                        
                        aug_label_name = f"{img_file.stem}_bal{i}.txt"
                        self.save_yolo_labels(self.balanced_label_dir / aug_label_name,
                                            transformed['bboxes'], transformed['class_labels'])
                    except Exception as e:
                        pass

        print("Balanceo completado.")

    def save_final_dataset(self):
        print("Guardando dataset final...")
        output_images_dir = PROCESSED_DATA_DIR / "images"
        output_labels_dir = PROCESSED_DATA_DIR / "labels"
        
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        valid_stems = set(p.stem for p in self.balanced_img_dir.glob("*")) & \
                      set(p.stem for p in self.balanced_label_dir.glob("*.txt"))

        print(f"✅ Total de pares válidos: {len(valid_stems)}")

        for stem in valid_stems:
            # Copy image
            for ext in ['.jpg', '.png', '.jpeg']:
                src_img = self.balanced_img_dir / (stem + ext)
                if src_img.exists():
                    shutil.copy2(src_img, output_images_dir / (stem + ext))
                    break
            
            # Copy label
            src_label = self.balanced_label_dir / (stem + ".txt")
            shutil.copy2(src_label, output_labels_dir / (stem + ".txt"))

        print(f"Dataset final guardado en {PROCESSED_DATA_DIR}")

    @staticmethod
    def read_yolo_labels(label_path):
        bboxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    coords = list(map(float, parts[1:]))
                    bboxes.append(coords)
                    labels.append(cls_id)
        return bboxes, labels

    @staticmethod
    def save_yolo_labels(label_path, bboxes, labels):
        with open(label_path, 'w') as f:
            for label, box in zip(labels, bboxes):
                f.write(f"{label} {' '.join(map(str, box))}\\n")

    def run(self):
        self.resize_images()
        self.normalize_images()
        self.augment_data()
        self.balance_classes()
        self.save_final_dataset()

if __name__ == "__main__":
    from src.config import FRAMES_VIDEO_DIR, ANNOTATIONS_YOLO_DIR
    preprocessor = Preprocessor(FRAMES_VIDEO_DIR, ANNOTATIONS_YOLO_DIR)
    preprocessor.run()
