import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TEMP_DATA_DIR = DATA_DIR / "temp"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
YOLO_DATA_DIR = DATA_DIR / "yolo"
COCO_DATA_DIR = DATA_DIR / "coco"

# Subdirectories
FRAMES_VIDEO_DIR = RAW_DATA_DIR / "frames_video"
ANNOTATIONS_YOLO_DIR = RAW_DATA_DIR / "annotations_yolo" / "obj_train_data"

# Processing parameters
TARGET_SIZE = (640, 640)
SEED = 42
