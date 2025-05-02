import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math

def compute_feature_stats(df_subset):
    stats = []
    for col in df_subset.columns:
        if col == 'class':
            continue
        col_values = df_subset[col]
        n_unique = col_values.nunique()
        unique_vals = sorted(col_values.unique())
        missing = col_values.isnull().sum()
        is_categorical = True
        stat = {
            "Feature": col,
            "Type": "Categorical",
            "Unique Values": unique_vals,
            "Missing": missing,
            "Outliers": "N/A"
        }
        stats.append(stat)
    return pd.DataFrame(stats)


def report_class_balance(df, target_col='class'):
    class_counts = df[target_col].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"Class {label} - Count: {count}")
    ratio = class_counts[1] / class_counts[0] if 0 in class_counts and 1 in class_counts else None
    if ratio is not None:
        print(f"Class imbalance ratio (1/0): {ratio:.2f}")
