from sklearn.model_selection import GroupShuffleSplit


def make_grouped_train_val_test_split(
    X,
    y,
    groups,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    if val_size <= 0 or test_size <= 0:
        raise ValueError("val_size and test_size must both be positive.")

    # First split: train vs holdout
    holdout_size = val_size + test_size
    first_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_size,
        random_state=random_state,
    )

    train_idx, holdout_idx = next(first_splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    groups_train = groups.iloc[train_idx].copy()

    X_holdout = X.iloc[holdout_idx].copy()
    y_holdout = y.iloc[holdout_idx].copy()
    groups_holdout = groups.iloc[holdout_idx].copy()

    # Second split: validation vs test inside holdout
    # Need the relative size of test within holdout
    relative_test_size = test_size / (val_size + test_size)

    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=random_state,
    )

    val_idx, test_idx = next(
        second_splitter.split(X_holdout, y_holdout, groups=groups_holdout)
    )

    X_val = X_holdout.iloc[val_idx].copy()
    y_val = y_holdout.iloc[val_idx].copy()
    groups_val = groups_holdout.iloc[val_idx].copy()

    X_test = X_holdout.iloc[test_idx].copy()
    y_test = y_holdout.iloc[test_idx].copy()
    groups_test = groups_holdout.iloc[test_idx].copy()

    # Sanity checks
    train_group_set = set(groups_train.unique())
    val_group_set = set(groups_val.unique())
    test_group_set = set(groups_test.unique())

    if train_group_set & val_group_set:
        raise ValueError("Leakage detected: overlapping VIDEO_IDs between train and val.")
    if train_group_set & test_group_set:
        raise ValueError("Leakage detected: overlapping VIDEO_IDs between train and test.")
    if val_group_set & test_group_set:
        raise ValueError("Leakage detected: overlapping VIDEO_IDs between val and test.")

    print("Grouped train/val/test split created successfully.")
    print(
        f"Rows -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    print(
        f"Videos -> train: {groups_train.nunique()}, "
        f"val: {groups_val.nunique()}, test: {groups_test.nunique()}"
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        groups_train, groups_val, groups_test,
    )