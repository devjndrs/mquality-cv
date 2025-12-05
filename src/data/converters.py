import os
import json
from pathlib import Path
from PIL import Image
from src.config import YOLO_DATA_DIR, COCO_DATA_DIR

# Categories configuration (should match data.yaml)
CATEGORIES = [
    {"id": 1, "name": "person"},
    # Add more classes as needed
]

class YoloToCocoConverter:
    def __init__(self, yolo_dir, output_dir):
        self.base_dir = Path(yolo_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert(self, splits=["train", "val"]):
        for split in splits:
            images_dir = self.base_dir / "images" / split
            labels_dir = self.base_dir / "labels" / split
            output_json = self.output_dir / f"coco_{split}.json"

            self._process_split(images_dir, labels_dir, output_json)

    def _process_split(self, images_dir, labels_dir, output_json):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": CATEGORIES
        }

        annotation_id = 0
        img_id = 0
        
        # Using sorted to ensure deterministic order
        for img_file in sorted(images_dir.glob("*")):
            if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue
                
            try:
                with Image.open(img_file) as im:
                    width, height = im.size
            except Exception as e:
                print(f"Error reading image {img_file}: {e}")
                continue

            coco_format["images"].append({
                "id": img_id,
                "file_name": img_file.name,
                "width": width,
                "height": height
            })

            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(float(parts[0]))
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])

                            x_min = (x_center - w/2) * width
                            y_min = (y_center - h/2) * height
                            w_px = w * width
                            h_px = h * height

                            coco_format["annotations"].append({
                                "id": annotation_id,
                                "image_id": img_id,
                                "category_id": cls + 1, # COCO is usually 1-indexed, YOLO 0-indexed. Adjust if needed.
                                "bbox": [x_min, y_min, w_px, h_px],
                                "area": w_px * h_px,
                                "iscrowd": 0
                            })
                            annotation_id += 1
            
            img_id += 1

        with open(output_json, "w") as f:
            json.dump(coco_format, f, indent=4)

        print(f"COCO JSON guardado en: {output_json}")

if __name__ == "__main__":
    converter = YoloToCocoConverter(YOLO_DATA_DIR, COCO_DATA_DIR)
    converter.convert()
