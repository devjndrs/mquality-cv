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

## 🛠️ Technologies & Tools

This project leverages a modern Data Engineering stack to ensure efficient and reliable processing of computer vision datasets:

### Orchestration & Workflow
*   **Apache Airflow**: Manages and schedules the complex workflows of the data pipeline, ensuring task dependencies, retries, and monitoring are handled robustly.

### Core Processing & Logic
*   **Python 3.10+**: The primary programming language for all pipeline logic.
*   **OpenCV & Pillow (PIL)**: High-performance libraries for image manipulation and processing.
*   **NumPy**: Fundamental package for numerical computations and array manipulation.

### Data Augmentation & Transformation
*   **Albumentations**: A fast and flexible library for image augmentation, critical for improving model robustness and generalizing datasets.

### Data Validation & Quality Assurance
*   **Pandas**: Used for generating validation reports, logs, and analyzing dataset statistics.
*   **Hashlib**: Implements data integrity checks to detect duplicate or corrupt files efficiently.

### Deep Learning & Formats
*   **PyTorch**: The underlying framework for model training and tensor operations.
*   **YOLO (Ultralytics)**: Object detection model architecture supported by the pipeline's output format.
*   **COCO API**: Standard format for object detection annotations, ensuring compatibility with various frameworks.

### Version Control
*   **Git**: Source code management.
*   **DVC (Data Version Control)**: *Recommended* for managing large dataset versions and pipeline reproducibility.

## License

[MIT License](LICENSE)
