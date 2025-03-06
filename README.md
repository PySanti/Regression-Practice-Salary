# Salary

El objetivo del proyecto sera crear un meta-modelo de stacking (en el contexto de ensemble learning) utilizando algoritmos de regresion, especificamente:

* Regresion Lineal
* SVR
* Random Forest

La idea sera encontrar la mejor combinacion de hiperparametros posibles para cada algoritmo para cada posible variante del dataset de entrenamiento. Se realizara un proceso de preprocesamiento basico y generar diferentes alternativas para preprocesamiento del dataset. Por ejemplo, con scaler, sin scaler, etc. Utilizaremos la **matriz de estudio**:


| **Variante de preprocesamiento** | **Algoritmo 1** | **Algoritmo 2** | **Algoritmo 3** |
|--------------------------------|-----------------|-----------------|-----------------|
| **Con PCA, sin scaler**        | Precisión 1     | Precisión 2     | Precisión 3     |
| **Con scaler, sin PCA**        | Precisión 4     | Precisión 5     | Precisión 6     |
| **Con PCA y scaler**           | Precisión 7     | Precisión 8     | Precisión 9     |
| **Sin PCA ni scaler**          | Precisión 10    | Precisión 11    | Precisión 12    |


## Preprocesamiento

1- Manejo de valores Nan: las columnas que contienen valores Nan son las siguientes.

columna             porcentaje de valores nan
______________________________
Age                 0.029%\
Gender              0.029%\
Education Level     0.044%\
Job Title           0.029%\
Years of Experience 0.044%\
Salary              0.074%



2- Codificacion: las columnas categoricas son las siguientes.

columna             categorias
______________________________
Gender              3\
Education Level     7\
Job Title           193

Se puede considerar utilizar One-Hot-Encoding para las columnas con menos categoricas y Target-Encoding para las demas.

3- Scalers.

4- Estudio de correlaciones.

5- Estudio de distribuciones gaussianas: es imposible en un problema de regresion.

6- Extraccion y/o seleccion de caracteristicas.

7- Desequilibrio de datos.


## Entrenamiento

Finalmente se opto por no usar `SVR` dada la exigencia computacional que exige.

| Variante de preprocesamiento | Regresion Lineal                                                                                                                             | SVR | Random Forest                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-----|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Scaler +; PCA +              | Mejores hiperparametros<br>{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}<br>Mejor MAE<br>17353.6                | -   | Mejores hiperparametros<br>{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(400), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}<br>Mejor MAE<br>2721.832178267951 |
| Scaler +; PCA -              | Mejores hiperparametros<br>{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}<br>Mejor MAE<br>17342.6                | -   | Mejores hiperparametros<br>{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 20, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(350), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}<br>Mejor MAE<br>2572.0913234724912  |
| Scaler -; PCA +              | Mejores hiperparametros<br>{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': True}<br>Mejor MAE<br>40613.2                 | -   | Mejores hiperparametros<br>{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(400), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}<br>Mejor MAE<br>3149.291180832718   |
| Scaler -; PCA -              | Mejores hiperparametros<br>{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}<br>Mejor precision<br>17342.6860153148 | -   | Mejores hiperparametros<br>{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 30, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(200), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}<br>Mejor MAE<br>2565.08580718767    |
