import numpy as np
import pandas as pd

from config import PROCESSED_DATA_PATH
from feature_engineering import load_processed_data, build_model_dataset

df = load_processed_data(PROCESSED_DATA_PATH)
X, y, groups, feature_cols = build_model_dataset(df)

print("Feature columns:")
print(feature_cols)
print()

print("NaN counts:")
print(X.isna().sum())
print()

print("Infinite counts:")
print(np.isinf(X).sum())
print()

print("Very large absolute values:")
print((X.abs() > 1e10).sum())
print()

print("Summary:")
print(X.describe().T)