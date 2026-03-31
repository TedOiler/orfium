from config import PROCESSED_DATA_PATH
from feature_engineering import load_processed_data, build_model_dataset
from split import make_grouped_train_test_split

df = load_processed_data(PROCESSED_DATA_PATH)
X, y, groups, feature_cols = build_model_dataset(df)

print(X.shape, y.shape, groups.shape)
print(feature_cols)

X_train, X_test, y_train, y_test, groups_train, groups_test = make_grouped_train_test_split(X, y, groups)

print(X_train.shape, X_test.shape)
print(groups_train.nunique(), groups_test.nunique())
print(set(groups_train).intersection(set(groups_test)))