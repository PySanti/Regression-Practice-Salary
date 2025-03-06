from utils.filename_gen import filename_gen
from utils.basic_preprocess import basic_preprocess
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np



TARGET = "Salary"
param_grid = {
    'n_estimators': np.arange(50, 401, 50),         # Número de árboles en el bosque
    'max_depth': [None, 10, 20, 30, 40],         # Máxima profundidad de los árboles
    'min_samples_split': [2, 5, 10],             # Número mínimo de muestras requeridas para dividir un nodo
    'min_samples_leaf': [1, 2, 4],               # Número mínimo de muestras requeridas en una hoja
    'max_features': ['sqrt', 'log2'],    # Número de características a considerar al buscar la mejor división
    'bootstrap': [True, False]                   # Si utilizar o no bootstrap samples al construir árboles
}


for scaler, pca in [(True, True), (True, False), (False, True), (False, False)]:
    filename = filename_gen("rf", scaler, pca)
    print(filename)
    [df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), TARGET, scaler=scaler, pca=pca)

    grid_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_grid, cv=4, verbose=10,n_jobs=6, scoring="neg_mean_absolute_error", n_iter=300)
    grid_search.fit(df_train.drop(TARGET, axis=1), df_train[TARGET])


    print("Mejores hiperparametros")
    print(grid_search.best_estimator_.get_params())
    print("Mejor MAE")
    print(grid_search.best_score_*-1)

    joblib.dump(grid_search.best_estimator_, filename)


