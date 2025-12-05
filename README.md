# MQuality CV - Computer Vision Data Engineering Pipeline

## Overview

MQuality CV is a robust Data Engineering pipeline designed for Computer Vision tasks. It automates the end-to-end process of data validation, preprocessing, augmentation, and formatting for object detection models (YOLO, COCO). This project emphasizes modularity, reproducibility, and scalability, adhering to best practices in software engineering and data operations (DataOps).

## Project Structure

The project is organized into a modular architecture to separate concerns and facilitate maintenance:

```
mquality-cv/
├── airflow-dags/       # Airflow DAGs for workflow orchestration
├── data/               # Data storage (raw, temp, processed, yolo, coco)
├── notebooks/          # Experimental notebooks
├── reports/            # Validation reports and logs
├── src/                # Source code
│   ├── data/           # Data processing modules
│   │   ├── converters.py   # Format conversion (YOLO -> COCO)
│   │   ├── dataset.py      # Dataset splitting and organization
│   │   └── preprocessing.py # Image resizing, normalization, augmentation
│   ├── models/         # Model training and definition
│   │   └── train.py        # Training entry point
│   ├── validation/     # Data validation scripts
│   │   ├── check_images.py # Image corruption and duplicate checks
│   │   └── check_labels.py # Annotation integrity checks
│   ├── config.py       # Configuration and constants
│   └── main.py         # Main pipeline execution entry point
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Pipeline Stages

The pipeline consists of the following stages, which can be executed independently or sequentially:

1.  **Validation**:
    *   **Image Validation**: Detects corrupt images and duplicates using hashing.
    *   **Label Validation**: Verifies consistency between images and labels, and checks bounding box coordinates.
    *   Generates reports in `reports/`.

2.  **Preprocessing**:
    *   **Resizing**: Standardizes image resolution (default: 640x640).
    *   **Normalization**: Normalizes pixel values.
    *   **Augmentation**: Applies geometric and color transformations using Albumentations.
    *   **Class Balancing**: Balances class distribution through oversampling.

3.  **Dataset Preparation**:
    *   Splits data into Training and Validation sets (default: 80/20).
    *   Organizes data into the standard YOLO directory structure.

4.  **Format Conversion**:
    *   Converts YOLO annotations to COCO JSON format for compatibility with various frameworks.

5.  **Training**:
    *   (Placeholder) Module ready for integrating model training logic (e.g., Ultralytics YOLO, PyTorch).

## Usage

The pipeline is controlled via the `src/main.py` script. Ensure you have the dependencies installed.

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Pipeline

You can run the entire pipeline or specific steps using the `--step` argument.

**Run all steps:**
```bash
python src/main.py --step all
```

**Run specific steps:**
```bash
# Run data validation
python src/main.py --step validate

# Run preprocessing
python src/main.py --step preprocess

# Prepare YOLO dataset structure
python src/main.py --step dataset

# Convert to COCO format
python src/main.py --step convert

# Run training
python src/main.py --step train
```

## Configuration

Key paths and parameters (e.g., directories, seed, target size) are defined in `src/config.py`. Modify this file to adapt the pipeline to your specific environment.

## Technologies

*   **Python**: Core language.
*   **OpenCV & PIL**: Image processing.
*   **Albumentations**: Data augmentation.
*   **PyTorch**: Deep learning framework support.
*   **Pandas**: Reporting and data manipulation.

## License

[MIT License](LICENSE)
