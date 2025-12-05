import argparse
import sys
from pathlib import Path

# Add src to python path to ensure imports work if run from root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import FRAMES_VIDEO_DIR, ANNOTATIONS_YOLO_DIR, PROCESSED_DATA_DIR, YOLO_DATA_DIR, COCO_DATA_DIR, DATA_DIR
from src.data.preprocessing import Preprocessor
from src.data.dataset import DatasetSplitter
from src.data.converters import YoloToCocoConverter
from src.models.train import train_model
from src.validation import check_labels, check_images

def run_validation():
    print("Running validation...")
    print("Running label check...")
    check_labels.run()
    print("Running image check...")
    check_images.run()

def run_preprocessing():
    print("Running preprocessing...")
    preprocessor = Preprocessor(FRAMES_VIDEO_DIR, ANNOTATIONS_YOLO_DIR)
    preprocessor.run()

def run_dataset_prep():
    print("Running dataset preparation (YOLO structure)...")
    splitter = DatasetSplitter(PROCESSED_DATA_DIR / "images", PROCESSED_DATA_DIR / "labels", YOLO_DATA_DIR)
    splitter.split_dataset()

def run_coco_conversion():
    print("Running YOLO to COCO conversion...")
    converter = YoloToCocoConverter(YOLO_DATA_DIR, COCO_DATA_DIR)
    converter.convert()

def run_training():
    print("Running training...")
    train_model(DATA_DIR)

def main():
    parser = argparse.ArgumentParser(description="MQuality CV Pipeline")
    parser.add_argument("--step", type=str, choices=["all", "validate", "preprocess", "dataset", "convert", "train"], default="all", help="Pipeline step to run")
    
    args = parser.parse_args()

    if args.step == "all" or args.step == "validate":
        run_validation()
    
    if args.step == "all" or args.step == "preprocess":
        run_preprocessing()
        
    if args.step == "all" or args.step == "dataset":
        run_dataset_prep()
        
    if args.step == "all" or args.step == "convert":
        run_coco_conversion()
        
    if args.step == "all" or args.step == "train":
        run_training()

if __name__ == "__main__":
    main()
