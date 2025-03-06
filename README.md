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

Age                 0.029% (porcentaje de valores nan)\
Gender              0.029%\
Education Level     0.044%\
Job Title           0.029%\
Years of Experience 0.044%\
Salary              0.074%



2- Codificacion: las columnas categoricas son las siguientes.

Gender              3 (categorias)\
Education Level     7\
Job Title           193

Se puede considerar utilizar One-Hot-Encoding para las columnas con menos categoricas y Target-Encoding para las demas.

3- Scalers: se utilizo como variante de preprocesamiento.

4- Estudio de correlaciones: no se estudiaron correlaciones.

5- Estudio de distribuciones gaussianas: es imposible en un problema de regresion.

6- Extraccion y/o seleccion de caracteristicas: no se hizo un proceso de seleccion de caracteristicas, en cambio, si se implemento PCA.

7- Desequilibrio de datos.


## Entrenamiento

Finalmente se opto por no usar `SVR` dada su exigencia computacional.

### Regresion Lineal

| Variante de preprocesamiento |                                                                                                                        Regresion Lineal                                                                                                                        |
|:----------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        Scaler +; PCA +       | Mejores hiperparametros: `{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}`<br>Mejor MAE para validacion : 17353.6<br>MAE para train : 17310.997<br>R2 para train : 0.814<br>MAE para test : 16990.632<br>R2 para test : 0.814       |
|        Scaler +; PCA -       | Mejores hiperparametros: {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}<br>Mejor MAE para validacion: 17342.6<br>MAE para train : 17288.520<br>R2 para train : 0.814<br>MAE para test : 16973.456<br>R2 para test : 0.815          |
|        Scaler -; PCA +       | Mejores hiperparametros: {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': True}<br>Mejor MAE para validacion: 40613.2<br>MAE para train : 40607.194<br>R2 para train : 0.167<br>MAE para test : 40303.456<br>R2 para test : 0.163           |
|        Scaler -; PCA -       | Mejores hiperparametros: {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}<br>Mejor MAE para validacion: 17342.6860153148<br>MAE para train : 17288.520<br>R2 para train : 0.814<br>MAE para test : 16973.456<br>R2 para test : 0.815 |


### Random Forest



| Variante de preprocesamiento |                                                                                                                                                                                                                                                                                                   Random Forest                                                                                                                                                                                                                                                                                                   |
|:----------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        Scaler +; PCA +       | Mejores hiperparametros: `{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(400), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}`<br>Mejor MAE para validacion: 2721.832178267951<br>MAE para train : 908.762<br>R2 para train : 0.997<br>MAE para test : 2893.290<br>R2 para test : 0.972 |
|        Scaler +; PCA -       | Mejores hiperparametros: `{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 20, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(350), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}`<br>Mejor MAE para validacion: 2572.0913234724912<br>MAE para train : 919.654<br>R2 para train : 0.997<br>MAE para test : 2704.861<br>R2 para test : 0.976  |
|        Scaler -; PCA +       | Mejores hiperparametros: `{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(400), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}`<br>Mejor MAE para validacion: 3149.291180832718<br>MAE para train : 911.033<br>R2 para train : 0.997<br>MAE para test : 4254.626<br>R2 para test : 0.923   |
|        Scaler -; PCA -       | Mejores hiperparametros: `{'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 30, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(200), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}`<br>Mejor MAE para validacion: 2565.08580718767<br>MAE para train : 915.605<br>R2 para train : 0.997<br>MAE para test : 2705.039<br>R2 para test : 0.976    |
