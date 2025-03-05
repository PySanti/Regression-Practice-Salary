from utils.filename_gen import filename_gen
from utils.basic_preprocess import basic_preprocess
import pandas as pd


TARGET = "Salary"


for scaler, pca in [(True, True), (True, False), (False, True), (False, False)]:
    print(filename_gen("svr", scaler, pca))
    [df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), TARGET, scaler=scaler, pca=pca)
    print(df_train.head(5))
    print(df_test.head(5))
    print(df_val.head(5))
    

