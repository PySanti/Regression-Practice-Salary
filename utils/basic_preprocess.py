from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess.nan_fixer import CustomImputer


def basic_preprocess(df, target : str):
    df_train, unseen_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    df_test, df_val = train_test_split(unseen_df, test_size=0.2, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("imputer", CustomImputer(strategy="mean", df.columns.tolist()))
    ])

    
    
