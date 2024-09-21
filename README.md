# Análisis de Precios de Casas en Boston

Este proyecto se centra en el análisis y modelado de precios de casas en Boston utilizando técnicas de regresión. El objetivo es desarrollar un modelo predictivo que permita estimar el precio de las viviendas en función de diversas características socioeconómicas y ambientales.

#Integrantes:
- Cima, Nancy Lucía - nancy.cima.bertoni@hotmail.com
- Longo, Gonzalo - longogonzalo.g@gmail.com
- Sumiacher, Julia - jsumiacher@gmail.com
  
## Contenido

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Datos Utilizados](#datos-utilizados)
- [Metodología](#metodología)
- [Modelos Implementados](#modelos-implementados)
- [Resultados](#resultados)
- [Conclusiones](#conclusiones)
- [Requisitos](#requisitos)
- [Ejecutar el Proyecto](#ejecutar-el-proyecto)

## Descripción del Proyecto

Este proyecto aborda la predicción de precios de casas en Boston utilizando un conjunto de datos que incluye variables como la tasa de criminalidad, el número promedio de habitaciones, la proporción de terrenos residenciales, entre otros. Se implementan diversas técnicas de regresión para identificar el modelo más adecuado.

## Datos Utilizados

El dataset contiene información sobre casas en Boston, incluyendo características socioeconómicas y ambientales que pueden influir en el precio de las viviendas. 

## Metodología

1. **Exploración de Datos**: Análisis descriptivo para entender la distribución de las variables y sus relaciones.
2. **Preprocesamiento**: Limpieza de datos y tratamiento de valores atípicos.
3. **Modelado**: Implementación de diversos modelos de regresión, incluyendo regresión lineal, Ridge, Lasso y ElasticNet.
4. **Evaluación**: Cálculo de métricas como R², RMSE y MAE para evaluar el rendimiento de los modelos.

## Modelos Implementados

- **Regresión Lineal**
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet**
- **Gradient Descent**
- **Stochastic Gradient Descent**
- **Mini-Batch Gradient Descent**

## Resultados

Los resultados muestran que los modelos de regularización, especialmente ElasticNet, ofrecen el mejor rendimiento, con un R² de 0.364 y un RMSE de 52.89 en el conjunto de prueba. La optimización de hiperparámetros demostró ser crucial para mejorar el ajuste del modelo.

## Conclusiones

El proyecto resalta la importancia de la selección del modelo y la optimización de hiperparámetros para mejorar el rendimiento predictivo. Los hallazgos tienen implicaciones significativas para el sector inmobiliario, ofreciendo herramientas para la toma de decisiones informadas.
