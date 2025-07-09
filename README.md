# Alzheimer's Disease Prediction from MRI Scans

This project implements a deep learning pipeline for predicting Alzheimer's disease severity from brain MRI scans. The pipeline includes data preprocessing, model training, evaluation, hyperparameter tuning, and visualization. The codebase is modular and supports both custom CNN and ResNet-style architectures.

## Project Structure

- `data_preprocessing.py`: Loading, preprocessing, and augmenting MRI data. Handles 2D and NIfTI images, metadata integration, and dataset splitting.
- `model.py`: Contains `AlzheimerCNN` and `AlzheimerResNet` architectures implemented in PyTorch.
- `train.py`: Training loop, validation, evaluation metrics, and model checkpointing.
- `augmentation.py`: Data augmentation transforms and visualization utilities.
- `hyperparameter_tuning.py`: Grid search for model/optimizer/lr/scheduler selection.
- `visualization.py`: Visualization of confusion matrix, ROC curve, and predictions.
- `inference.py`: Inference pipeline for new images (2D or NIfTI).
- `main.py`: Main script to run the full pipeline with command-line arguments.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- scikit-learn
- nibabel
- Pillow (PIL)
- OpenCV (optional, for visualization)
- pandas (for Excel metadata)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Data Preparation

- Place your MRI images in the `data/` directory, organized by class (e.g., `data/NonDemented/`, `data/MildDemented/`, etc.).
- Optionally, include an Excel file with patient metadata in the `data/` directory.

## Usage

### Training
Run the main pipeline with default settings:
```bash
python main.py --data_dir ./data --output_dir ./output --model resnet --epochs 20 --batch_size 32 --augment --gpu
```

### Hyperparameter Tuning
```bash
python main.py --tune
```

### Inference
```bash
python inference.py --model_path ./models/alzheimer_resnet.pth --image_path ./data/NonDemented/sample.png
```

## Methodology
- Data is preprocessed (resized, normalized, augmented) and split into train/val/test (80/10/10).
- Two models are available: a custom CNN and a ResNet-style model with residual blocks.
- Training uses Adam or SGD, cross-entropy loss, and optional learning rate scheduling.
- Grid search is available for hyperparameter optimization.
- Evaluation includes accuracy, precision, recall, F1-score, ROC AUC, and visualizations.

## Results
- The best model achieves high accuracy and robust performance on the test set.
- See `output/` for training history, metrics, and model checkpoints.

## Acknowledgements
- Developed by Sanjay Singh as part of an internship assignment.
- Supervised by Prof. Mallikharjuna Rao, IIIT Naya Raipur.
