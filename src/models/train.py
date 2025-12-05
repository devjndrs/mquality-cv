import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CocoDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Leer JSON COCO
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Guardar lista de imágenes
        self.imgs = coco_data["images"]
        self.annotations = coco_data["annotations"]

        # Mapear imagen_id → anotaciones
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_id = img_info["id"]

        # Cargar imagen
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Obtener anotaciones
        anns = self.img_id_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, w, h]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(data_dir, epochs=10):
    print(f"Starting training with data from {data_dir}...")
    # Placeholder for training logic
    # You can implement your training loop here or call an external library like Ultralytics YOLO
    
    # Example of loading data
    # transform = transforms.ToTensor()
    # dataset = CocoDataset(
    #     annotations_file=os.path.join(data_dir, "coco/coco_train.json"),
    #     img_dir=os.path.join(data_dir, "yolo/images/train"),
    #     transform=transform
    # )
    # loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    print("Training logic to be implemented.")

if __name__ == "__main__":
    from src.config import DATA_DIR
    train_model(DATA_DIR)
