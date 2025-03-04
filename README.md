
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

## Entrenamiento