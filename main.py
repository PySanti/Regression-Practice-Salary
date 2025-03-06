from utils.filename_gen import filename_gen
from utils.basic_preprocess import basic_preprocess
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

TARGET = "Salary"
param_list =[
    {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False},
    {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False},
    {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': True},
    {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}
]


i = 0
for scaler, pca in [(True, True), (True, False), (False, True), (False, False)]:

    filename = filename_gen("rf", scaler, pca)
    print(filename)

    [df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), TARGET, scaler=scaler, pca=pca)

    best_params = param_list[i]
    alg = LinearRegression(**best_params)
    alg.fit(df_train.drop(TARGET, axis=1), df_train[TARGET])

    # train
    y_pred = alg.predict(df_train.drop(TARGET, axis=1))
    print(f"MAE para train : {mean_absolute_error(df_train[TARGET], y_pred):.3f}")
    print(f"R2 para train : {r2_score(df_train[TARGET], y_pred):.3f}")

    # test
    y_pred = alg.predict(df_test.drop(TARGET, axis=1))
    print(f"MAE para test : {mean_absolute_error(df_test[TARGET], y_pred):.3f}")
    print(f"R2 para test : {r2_score(df_test[TARGET], y_pred):.3f}")


    i+=1
