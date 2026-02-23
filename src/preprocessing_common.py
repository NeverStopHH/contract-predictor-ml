import pandas as pd
import numpy as np

# These are the numeric columns that require cleaning due to
# currency symbols, non-breaking spaces and German decimal commas.
COLUMNS_TO_CLEAN = [
    "Quotedetail.Mietpreis",
    "Quotedetail.SummeNK",
    "Quotedetail.Heizkosten",
    "TurnoverRent",
    "TermofLeaseYears",
    "CreditRating",
]

LOG_TRANSFORM_COLS = [
    "Quotedetail.Mietpreis",
    "Quotedetail.SummeNK",
]


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans numeric columns by:
    - removing non-breaking spaces
    - removing currency symbols and non-numeric characters
    - converting German decimal commas to dots
    - casting values to float

    This ensures numerical consistency before modeling.
    """
    df = df.copy()

    for column in COLUMNS_TO_CLEAN:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(r'\xa0', '', regex=True)
            .str.replace(r'[^0-9,.-]', '', regex=True)
            .str.replace(',', '.')
            .astype(float)
        )

    return df


def encode_target(df: pd.DataFrame, target_col: str = "statecodenameQuote") -> pd.DataFrame:
    """
    Converts the categorical target variable into binary format:
    - 'Gewonnen' -> 1
    - 'Geschlossen' -> 0
    """
    df = df.copy()
    df[target_col] = df[target_col].map({'Gewonnen': 1, 'Geschlossen': 0})
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "statecodenameQuote"):
    """
    Separates features and target variable.
    """
    df = df.copy()
    y = df.pop(target_col)
    X = df
    return X, y


def apply_train_iqr_filter(train_X, train_y, test_X, test_y):
    """
    Applies IQR-based outlier filtering using the training distribution.
    The same boundaries are then applied to the test data.

    IQR factor = 2.0 (consistent with original modeling approach).
    """
    for col in COLUMNS_TO_CLEAN:
        Q1 = train_X[col].quantile(0.25)
        Q3 = train_X[col].quantile(0.75)
        IQR = Q3 - Q1

        train_valid = (train_X[col] >= Q1 - 2.0 * IQR) & (train_X[col] <= Q3 + 2.0 * IQR)
        test_valid = (test_X[col] >= Q1 - 2.0 * IQR) & (test_X[col] <= Q3 + 2.0 * IQR)

        train_X = train_X[train_valid]
        train_y = train_y[train_valid]
        test_X = test_X[test_valid]
        test_y = test_y[test_valid]

    return train_X, train_y, test_X, test_y


def log_transform(train_X, test_X):
    """
    Applies logarithmic transformation (log1p) to selected skewed variables
    to stabilize variance and reduce the impact of extreme values.
    """
    train_X = train_X.copy()
    test_X = test_X.copy()

    for col in LOG_TRANSFORM_COLS:
        train_X[col] = np.log1p(train_X[col])
        test_X[col] = np.log1p(test_X[col])

    return train_X, test_X
