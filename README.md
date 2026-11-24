# Dual-Level-Parallel-Isolation-Forest-for-Scalable-Blockchain-Anomaly-Detection
An experimental framework that explores parallel Isolation Forest approaches for blockchain anomaly detection across different dataset scales.

## Overview
This project investigates the application of Isolation Forest and parallel processing techniques for detecting anomalous patterns in blockchain market data. It evaluates the approach across two dataset scales to examine its detection performance, computational efficiency, and resource utilization.

The framework is organized into two evaluation environments:
- **Small-Scale Evaluation** – Uses a dataset containing 100,000 records.
- **Large-Scale Evaluation** – Uses a dataset containing 5,000,000 records.

Dataset Source: [Kaggle]([https://github.com](https://www.kaggle.com/datasets/mczielinski/Bitcoin-historical-data))

Both evaluations examine anomaly detection performance and computational resource requirements under different execution settings.

## Key Features
- **Unsupervised Anomaly Detection** – Applies Isolation Forest to identify potentially anomalous data points.
- **Dual-Level Evaluation** – Evaluates the framework using small- and large-scale datasets.
- **Parallel Processing** – Explores parallel execution to support computational scalability.
- **Ensemble-Based Detection** – Combines model outputs for anomaly identification.
- **Performance Evaluation** – Assesses detection performance using precision, recall, and F1-score.
- **Resource Monitoring** – Tracks execution time and memory consumption during experiments.

## Technologies Used
| Technology   | Purpose |
| ------------ | ------- |
| Python       | Core programming language for model implementation |
| Scikit-learn | Isolation Forest implementation and data preprocessing |
| Pandas       | Dataset loading, handling, and manipulation |
| NumPy        | Numerical computations and array operations |
| Joblib       | Parallel processing and computational efficiency |
| Psutil       | System resource and performance monitoring |

## Experimental Evaluation
The framework is evaluated using two dataset configurations to examine the relationship between dataset size, anomaly detection performance, and computational efficiency.

**Evaluation metrics include:**
- Precision
- Recall
- F1-score
- Execution time
- Memory consumption

## Published Research Paper
This repository accompanies a research paper published through the Institute of Electrical and Electronics Engineers (IEEE).

**Read the full paper:** [Research]([https://ieeexplore.ieee.org/document/11468242))

## Disclaimer
This repository showcases selected core implementations and major functions from the research project for portfolio and demonstration purposes. It does not contain the complete research codebase or all supporting components. The shared files highlight key methodologies and functionalities developed during the study.



