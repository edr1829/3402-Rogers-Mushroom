# Mushroom Classification – Kaggle Project

* **One Sentence Summary**  
This project uses a neural network built with Keras to predict whether a mushroom is edible or poisonous using categorical data from the Kaggle Mushroom Classification dataset.

## Overview

This challenge, as defined by the Kaggle competition at [Kaggle - Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification), is to determine if a mushroom is edible 'e' or poisonous 'p' based on a set of physical attributes.  
Our approach treats this as a binary classification problem, using a Keras feedforward neural network trained on a one-hot encoded dataset.

Our final model achieved 99.2% accuracy on the validation set and showed strong generalization performance on the test set.

## Summary of Workdone

### Data

* Type: CSV file containing 22 categorical features and 1 target 'class'
* Size:
  * Rows: 8124
  * Features: 22 input columns → reduced to 9 selected + one-hot encoded
* Split:
  * 60% training, 20% validation, 20% test

#### Preprocessing / Clean up

- Selected 9 high-signal categorical features using visual separation and distribution analysis.
- Confirmed no missing values.
- All features label-encoded, then one-hot encoded.
- Applied MinMaxScaler to scale input values to (0,1) for neural network compatibility.

#### Data Visualization

- Histograms compared distributions of features between edible and poisonous classes.
- Used KS statistics and manual inspection.
- Key predictive features: bruises, odor, gill_size, gill_color, stalk-root, stalk-surface-above-ring, stalk-survace-below-ring, ring-type, spore-print-color

### Problem Formulation

* Input: One-hot encoded version of 9 selected categorical features
* Output: Binary prediction of class 
* Model: Sequential() dense neural net with:
  * Layers: [12, 8, 8, 1]
  * Activation: ReLU (hidden), Sigmoid (output)
* Loss: Binary Crossentropy  
* Optimizer: Adam

### Training

* Framework: TensorFlow / Keras
* Training time: ~50 epochs with early stopping
* Validation accuracy: ~99.2%
* Batch size: 10
* Training curves monitored via history.history

### Performance Comparison

| Metric      | Validation |
|-------------|------------|
| Accuracy    | ~99.2%     |
| Precision   | High       |
| Recall      | High       |
| Confusion Matrix | True positives and negatives both high |

### Conclusions

- The selected features carry extremely strong signal.
- Neural networks perform excellently on categorical data when properly encoded.
- No class imbalance issues impacted performance.

### Future Work

- Try XGBoost and compare to Keras performance.
- Perform feature selection and ablation studies.
- Add hyperparameter tuning (e.g., with KerasTuner or GridSearch).

## How to reproduce results

### Run the notebook:

```bash
jupyter notebook "3402 Rogers Mushroom.ipynb"
```

Ensure you have:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib kagglehub
```

### Dataset:

Automatically downloaded via:
```python
import kagglehub
path = kagglehub.dataset_download("uciml/mushroom-classification")
```

### Training:

Run all cells through the notebook. Training occurs with:
```python
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
```

#### Performance Evaluation:

```python
from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_val, model.predict(X_val) > 0.5))
```

## Overview of files in repository

* `3402 Rogers Mushroom.ipynb` — Main project notebook
* `dataloader_mushroom.py` — Handles kagglehub download and initial loading
* `eda_tools.py` — Functions for comparing distributions and stats
* `submission.csv` — Output file for Kaggle

### Software Setup

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn kagglehub
```

## Citations

- UCI Mushroom Classification Dataset (via Kaggle): https://www.kaggle.com/datasets/uciml/mushroom-classification
- TensorFlow/Keras documentation: https://www.tensorflow.org/
- University of Texas at Arlington, Data Science Course Materials (DATA 3402)
