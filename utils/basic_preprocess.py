from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess.nan_fixer import CustomImputer
from preprocess.encoding import FrecuencyEncoding, CustomOneHotEncoding
from sklearn.decomposition import PCA
from preprocess.scaler import CustomScaler
import pandas as pd


def basic_preprocess(df, target : str, scaler=False, pca=False):


    df_train, unseen_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    df_test, df_val = train_test_split(unseen_df, test_size=0.5, shuffle=True, random_state=42)

    pipe_basic = Pipeline([
        ("imputer", CustomImputer(strategy="most_frequent", attributes=df.columns.tolist())),
        ("freq-encoding", FrecuencyEncoding(["Job Title"])),
        ("one-hot-encoding", CustomOneHotEncoding(["Gender", "Education Level"])),
    ])

    df_train = pipe_basic.fit_transform(df_train, df_train[target])
    df_test = pipe_basic.transform(df_test)
    df_val = pipe_basic.transform(df_val)

    if scaler or pca:
        steps = []
        if scaler:
            steps.append(("scaler", CustomScaler(["Age", "Job Title", "Years of Experience"])))
        if pca:
            steps.append(("pca", PCA(n_components=0.99)))
        pipe_adv = Pipeline(steps)
        X_train    = pd.DataFrame(pipe_adv.fit_transform(df_train.drop(target, axis=1), df_train[target]), index=df_train.index)
        X_test     = pd.DataFrame(pipe_adv.transform(df_test.drop(target, axis=1)), index=df_test.index)
        X_val      = pd.DataFrame(pipe_adv.transform(df_val.drop(target, axis=1)), index=df_val.index)

        df_train    = pd.concat([X_train, df_train[target]], axis=1)
        df_test     = pd.concat([X_test, df_test[target]], axis=1)
        df_val      = pd.concat([X_val, df_val[target]], axis=1)


    return [df_train, df_test, df_val]

    
    
