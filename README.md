# K-Nearest Neighbors (KNN) for CIFAR-10 Image Classification

## Project Overview

This project contains the final implementation of a CIFAR-10 classification task using the K-Nearest Neighbors (KNN) algorithm. It implements three different data splitting and validation methods:

1.  **Train/Test Split**: Splits the entire dataset into training and test sets to evaluate the model.
2.  **Train/Validation/Test Split**: Further splits the training data into training and validation sets, uses the validation set to determine the best hyperparameter (e.g., k value), and evaluates final performance on the test set.
3.  **5-fold Cross-Validation**: Divides the dataset into 5 folds and performs training/testing five times to obtain a more stable generalization performance estimate.

For each experiment, accuracy, precision, recall, and F1-score are computed and reported. For cross-validation, performance variation with respect to the k value is visualized as a graph.

## Key Features

-   **Data Preprocessing**: Supports image resizing, grayscale conversion, and feature standardization.
-   **Dimensionality Reduction**: Optionally uses Incremental PCA to speed up computation.
-   **Three Experiment Modes**:
    1.  Simple Train/Test evaluation
    2.  Hyperparameter search using validation set and final testing
    3.  Stable evaluation using 5-fold cross-validation
-   **Multiple Performance Metrics**: Accuracy, precision, recall, F1-score
-   **Result Visualization**: Saves error bar plots showing accuracy change with respect to k in cross-validation.
-   **Modular Code Structure**: Each major functionality is organized into clear functions for readability and easy modification.

## Getting Started

### Environment Setup

It is recommended to use a virtual environment (e.g., venv, conda). Install required libraries using:

pip install -r requirements.txt

## How to Run

Run the main.py script via terminal or command prompt.

### Execution Examples

1.  **Run all experiment modes (default)**
    python main.py

2.  **Run 5-Fold Cross-Validation only**
    python main.py --mode cv

3.  **Specify custom dataset paths**
    python main.py --train_dir "D:\datasets\cifar-10\train" --labels_path "D:\datasets\cifar-10\trainLabels.csv"


## Output Results

After execution, all results will be saved in the OUT_DIR directory.

-   `train_test_results.csv`: Performance metrics for Train/Test split
-   `train_val_test_result.csv`: Optimal k and final test performance from validation split
-   `cv_summary.csv`: Mean and standard deviation metrics for each k in cross-validation
-   `cv_accuracy_vs_k.png`: Visualization of cross-validation accuracy vs k
