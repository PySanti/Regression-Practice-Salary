from utils.basic_preprocess import basic_preprocess
import pandas as pd


TARGET = "Salary"

[df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), TARGET)


