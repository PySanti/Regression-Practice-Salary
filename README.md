
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
Age                 0.029832935560859187%
Gender              0.029832935560859187%
Education Level     0.04474940334128878%
Job Title           0.029832935560859187%
Years of Experience 0.04474940334128878%
Salary              0.07458233890214797%



2- Codificacion: las columnas categoricas son las siguientes.

columna             categorias
______________________________
Gender              3
Education Level     7
Job Title           193

Se puede considerar utilizar One-Hot-Encoding para las columnas con menos categoricas y Target-Encoding para las demas.

3- Scalers.

4- Estudio de correlaciones.

5- Estudio de distribuciones gaussianas: es imposible en un problema de regresion.

6- Extraccion y/o seleccion de caracteristicas.

7- Desequilibrio de datos.


## Entrenamiento
