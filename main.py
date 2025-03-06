from utils.filename_gen import filename_gen
from utils.basic_preprocess import basic_preprocess
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

TARGET = "Salary"
param_list =[
    {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(400), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False},
    {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 20, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(350), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False},
    {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(400), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False},
    {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 30, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(200), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
]


i = 0
for scaler, pca in [(True, True), (True, False), (False, True), (False, False)]:

    filename = filename_gen("rf", scaler, pca)
    print(filename)

    [df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), TARGET, scaler=scaler, pca=pca)

    best_params = param_list[i]
    print(best_params)
    alg = RandomForestRegressor(**best_params)
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
