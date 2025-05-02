import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math
from sklearn.preprocessing import LabelEncoder

### load
path = kagglehub.dataset_download("uciml/mushroom-classification")
df = pd.read_csv(f"{path}/mushrooms.csv")

### encode
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

### analyze metadata
n_train = df.shape[0]
n_features = df.shape[1] - 1
any_null = np.any(df.isnull())

target_col = "class"
target_values = df[target_col]
n_unique = len(np.unique(target_values))

binary_classification = n_unique == 2
multiclass_classification = n_unique < n_train / 10 and not binary_classification
regression = not binary_classification and not multiclass_classification

dataset_summary = {
    "mushroom": [
        "mushroom",
        binary_classification,
        multiclass_classification,
        regression,
        n_train,
        any_null,
        n_features,
        n_features,
        0
    ]
}

def load_dataset(name):
    if name != "mushroom":
        raise ValueError("Only mushroom dataset is supported in this demo.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return {
        "Training": X,
        "Target": y,
        "Categorical": [True]*X.shape[1],
        "Attributes": list(X.columns),
        "Encoders": label_encoders
    }



