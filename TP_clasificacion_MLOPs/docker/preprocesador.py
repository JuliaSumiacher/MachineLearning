# Funcion similar a preprocesamiento en mlops.py pero sin tratar la variable target y

# Importamos librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
import joblib

def preprocesamiento(data):
  #Cargamos el dataset
  X = data.copy()
  
  #Antes que nada vemos si tenemos una fila completa de valores faltantes
  X.isna().all(axis=1).sum()
  
  # Eliminamos las columnas 'Evaporation' y 'Sunshine'
  X.drop(columns= ['Evaporation', 'Sunshine'], inplace = True)
  
  #Convertimos el tipo de dato de Date a datetime
  X['Date'] = pd.to_datetime(X['Date'])
  X['Date']
  
  # Agregamos una columna nueva con el mes de cada fecha
  X['Month'] = X['Date'].dt.month
  
  #Quitamos Date del conjunto porque ya no lo necesitamos.
  X.drop(['Date'], axis=1, inplace=True)

  # Seleccionamos columnas numéricas y categóricas
  numeric_columns = X.select_dtypes(include=['float64', 'int32', 'int64']).columns
  categorical_columns = X.select_dtypes(include=['object']).columns
  
  # IMPUTACION
  X.drop(columns = 'Location', axis = 1, inplace=True)
  
  X = X.reset_index(drop=True)
  
  # Función auxiliar para determinar si una variable numérica tiene muchos outliers
  def tiene_outliers(serie):
      """
      Determina si una serie numérica tiene muchos outliers.
      El criterio se basa en que si más del 5% de sus valores son outliers,
      la serie posee una gran cantidad de ouliers.
  
      Args:
      serie (pd.Series): La serie numérica a evaluar.
  
      Returns:
      bool: True si la serie tiene muchos outliers, False de lo contrario.
      """
      Q1 = serie.quantile(0.25)
      Q3 = serie.quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      outliers = serie[(serie < lower_bound) | (serie > upper_bound)]
  
      # Consideramos que tiene muchos outliers si más del 5% de los datos son outliers
      return len(outliers) / len(serie) > 0.05
  
  # Función para aplicar mediana o media a columnas numéricas y moda a las no numéricas
  def imputar_faltantes_por_mes(df, columna_mes, columnas, columnas_numericas, columnas_categoricas):
      """
      Imputa los valores faltantes de las columnas especificadas en un DataFrame agrupando por mes,
      utilizando la mediana, media o moda según el tipo de variable.
  
      Args:
      df (pd.DataFrame): El DataFrame con los datos.
      columna_mes (str): El nombre de la columna que contiene los meses.
      columnas (list): Lista de columnas a imputar.
      columnas_numericas (list): Lista de columnas numéricas.
      columnas_categoricas (list): Lista de columnas categóricas.
  
      Returns:
      pd.DataFrame: DataFrame con las columnas imputadas.
      """
  
      # Imputación por cada columna
      for columna in columnas_categoricas:
          # Imputar con la moda agrupando por mes
          df[columna] = df.groupby(columna_mes)[columna].transform(lambda grupo: grupo.fillna(grupo.mode()[0] if not grupo.mode().empty else np.nan))
  
      for columna in columnas_numericas:
          for mes, grupo in df.groupby(columna_mes):
              if tiene_outliers(grupo[columna].dropna()):  # Si tiene muchos outliers, usar la mediana
                  df.loc[df[columna_mes] == mes, columna] = grupo[columna].fillna(grupo[columna].median())
              else:  # De lo contrario, usar la media
                  df.loc[df[columna_mes] == mes, columna] = grupo[columna].fillna(grupo[columna].mean())
  
      return df
  
  # Aplicamos la función a cada conjunto de datos
  columnas_a_imputar = X.columns[X.isnull().sum() > 0]
  categorical_columns = categorical_columns.drop('Location')
  
  X = imputar_faltantes_por_mes(X, 'Month', columnas_a_imputar, numeric_columns, categorical_columns)
  
  # Quitamos las columnas categóricas mes agregadas para realizar la imputacion
  X.drop(columns = 'Month', axis = 1, inplace=True)
  
  numeric_columns = numeric_columns.drop('Month')
  
  # Codificación de variables categóricas

  #Codificación RainToday
  X['RainToday'] = X['RainToday'].map({'Yes': 1,  'No':0})
  
  def num_dir(df, n):
    """Recibe un dataframe y un numero de direcciones de viento
    y reemplaza en las columnas correspondientes a direcciones de viento
    segun sea el n"""
    data = df.copy(deep=True)
    cols = data.select_dtypes(include=['object']).columns
    if n == 4:
      data[cols] = data[cols].replace(['NW','NNE','NE','NNW'], 'N')
      data[cols] = data[cols].replace(['WNW','WSW'], 'W')
      data[cols] = data[cols].replace(['ENE','ESE'], 'E')
      data[cols] = data[cols].replace(['SW','SSW','SE','SSE'], 'S')
    elif n == 8:
      data[cols] = data[cols].replace(['NNE','NNW'], 'N')
      data[cols] = data[cols].replace(['WNW','WSW'], 'W')
      data[cols] = data[cols].replace(['ENE','ESE'], 'E')
      data[cols] = data[cols].replace(['SSW','SSE'], 'S')
    return data
  
  #Modificamos las direcciones de viento para cada conjunto
  #para tener 4 direcciones en vez de 16
  X = num_dir(X, 4)
  
  #Utilizamos el metodo get_dummies para hacer un one hot encoding estandar
  X = pd.get_dummies(X, prefix_sep='_', dtype=int)
  
  # Creamos el escalador
  scaler = RobustScaler()
  
  # Aplicamos el escalado solo a las columnas numéricas
  X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
  
  return X