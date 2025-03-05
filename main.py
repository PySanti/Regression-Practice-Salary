from utils.filename_gen import filename_gen
from utils.basic_preprocess import basic_preprocess
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import joblib


TARGET = "Salary"

param_grid = {
    'fit_intercept': [True, False], 
    'copy_X': [True, False],           
    'positive': [True, False]          
}

for scaler, pca in [(True, True), (True, False), (False, True), (False, False)]:
    filename = filename_gen("lr", scaler, pca)
    print(filename)
    [df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), TARGET, scaler=scaler, pca=pca)

    grid_search = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, verbose=10,n_jobs=5, scoring="neg_mean_absolute_error")
    grid_search.fit(df_train.drop(TARGET, axis=1), df_train[TARGET])


    print("Mejores hiperparametros")
    print(grid_search.best_estimator_.get_params())
    print("Mejor precision")
    print(grid_search.best_score_*-1)

    joblib.dump(grid_search.best_estimator_, filename)


