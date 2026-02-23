import pandas as pd


def one_hot_align(train_X: pd.DataFrame, test_X: pd.DataFrame, categorical_cols: list[str]):
    """
    One-hot encodes categorical columns and aligns the test feature space
    to the train feature space.
    """
    train_X = pd.get_dummies(train_X, columns=categorical_cols)
    test_X = pd.get_dummies(test_X, columns=categorical_cols)

    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

    return train_X, test_X