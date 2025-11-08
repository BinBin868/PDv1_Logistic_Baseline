import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def infer_columns(df: pd.DataFrame, target: str):
    cols=[c for c in df.columns if c!=target]
    num=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat=[c for c in cols if c not in num]
    return num, cat

def make_baseline_pipeline(num_cols, cat_cols) -> Pipeline:
    num = Pipeline([("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())])
    cat = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    pre = ColumnTransformer([("num",num,num_cols),("cat",cat,cat_cols)])
    clf = LogisticRegression(max_iter=200)
    return Pipeline([("prep",pre),("clf",clf)])
