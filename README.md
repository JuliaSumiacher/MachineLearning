# **Predicción de Lluvia en Australia - Aprendizaje Automático 1**

---

### **Integrantes**  

- **Cima, Nancy Lucía**  
  Email: [nancy.cima.bertoni@hotmail.com](mailto:nancy.cima.bertoni@hotmail.com)  
- **Longo, Gonzalo**  
  Email: [longogonzalo.g@gmail.com](mailto:longogonzalo.g@gmail.com)  
- **Sumiacher, Julia**  
  Email: [jsumiacher@gmail.com](mailto:jsumiacher@gmail.com)  

---
<div align="center">
  <img src="https://services.meteored.com/img/article/lluvias-e-inundaciones-historicas-en-zonas-de-australia-1646462914667_1024.gif" alt="Predicción de lluvia" width="600">
</div> 

### **Descripción del Proyecto**  
Este trabajo práctico tiene como objetivo principal predecir si lloverá al día siguiente en diferentes ciudades de Australia basándonos en datos climáticos históricos. La solución abarca el análisis exploratorio de datos, la implementación de modelos de clasificación, la optimización de hiperparámetros y la puesta en producción del modelo más adecuado utilizando herramientas modernas como Scikit-learn, TensorFlow y Docker.  

---

### **Objetivos del Trabajo Práctico**  

1. **Familiarización con herramientas clave:**  
   - Scikit-learn: Preprocesamiento, clasificación, métricas y explicabilidad de modelos.  
   - TensorFlow: Entrenamiento de redes neuronales.  
   - Docker: Puesta en producción.  

2. **Predicción de la variable `RainTomorrow`:**  
   - Predecir si lloverá al día siguiente en base a datos históricos de clima disponibles en el dataset `weatherAUS.csv`.  

3. **Análisis comparativo de modelos:**  
   - Comparar diferentes enfoques de clasificación para determinar el modelo más eficiente según las métricas seleccionadas.  

4. **Automatización y producción:**  
   - Implementar un modelo funcional y robusto para uso práctico en entornos reales.

---

### **Dataset**  

- **Nombre:** `weatherAUS.csv`  
- **Descripción:** Contiene información climática histórica de Australia, con múltiples variables que incluyen temperatura, humedad, presión, viento, etc. La columna objetivo es `RainTomorrow`, que indica si llovió o no al día siguiente.  
- **Procesamiento inicial:**  
  - Seleccionar aleatoriamente 10 ciudades de la columna `Location` y trabajar exclusivamente con los datos correspondientes a estas ciudades.  

---

### **Consignas**  

1. **Análisis Exploratorio de Datos (EDA):**  
   - Identificar y gestionar datos faltantes.  
   - Visualizar variables mediante histogramas, scatterplots y diagramas de caja.  
   - Evaluar el balanceo del dataset y justificar decisiones de preprocesamiento.  
   - Estandarizar/escalar datos según sea necesario.  
   - Generar una matriz de correlación para identificar relaciones entre variables.  

2. **Modelos de Clasificación:**  
   - Implementar regresión logística y evaluar métricas como accuracy, precision, recall y F1-score.  
   - Graficar matrices de confusión y curvas ROC, y analizar los umbrales adecuados.  

3. **Modelo Base y Optimización:**  
   - Crear un modelo base.  
   - Optimizar hiperparámetros utilizando técnicas como grid search, random search u Optuna.  

4. **Explicabilidad del Modelo:**  
   - Aplicar técnicas como SHAP para analizar variables a nivel local y global.  

5. **AutoML con Scikit-learn:**  
   - Probar la automatización de modelos y comparar resultados.  

6. **Redes Neuronales:**  
   - Implementar un modelo con TensorFlow y evaluar métricas.  
   - Comparar el rendimiento con la regresión logística.  

7. **Comparación de Modelos:**  
   - Evaluar todos los modelos utilizando métricas comunes y seleccionar el más adecuado.  

8. **MLOps y Producción:**  
   - Implementar la puesta en producción del modelo final utilizando Streamlit y Docker.  

9. **Conclusiones:**  
   - Documentar aprendizajes, limitaciones y resultados del trabajo.  

---

### **Requisitos Técnicos**  

#### **Dependencias:**  
- **Python:** Versión 3.8 o superior.  
- **Bibliotecas:**  
  - `scikit-learn`  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `seaborn`  
  - `tensorflow`  
  - `shap`  
  - `optuna`  
  - `pycaret`
  - `docker`  

---

### **Configuración Inicial:**  
1. Clonar este repositorio.  
2. Instalar dependencias con:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Asegurarse de tener instalado Docker.  


---

### **Puesta en Producción**  

#### **Instrucciones para Correr Docker**  

1. Construir el contenedor Docker:  
   ```bash
   docker build -t inference-python-test ./dockerfile
   ```  

2. Ejecutar el contenedor:  
   ```bash
   docker run -it --rm --name inference-python-test -v ./files:/files inference-python-test
   ```  

---

### **Objetivos**  
El proyecto cubre una amplia gama de actividades relacionadas con el aprendizaje automático, incluyendo:  

1. **Análisis Exploratorio de Datos (EDA):**  
   - Gestión de datos faltantes.  
   - Visualización y análisis de variables.  
   - Estandarización y escalado.  

2. **Implementación de Modelos:**  
   - Regresión logística.  
   - Optimización de hiperparámetros.  
   - Implementación de redes neuronales con TensorFlow.  

3. **Explicabilidad de Modelos:**  
   - Uso de SHAP para interpretar modelos.  

4. **Comparación de Modelos:**  
   - Evaluación y selección del mejor modelo.  

5. **Producción del Modelo:**  
   - Puesta en producción con Streamlit y Docker.  

---

### **Conclusiones y Reflexión Final**  
Este trabajo práctico permitió a los integrantes profundizar en técnicas modernas de clasificación, análisis de datos y despliegue de modelos en producción, fortaleciendo habilidades críticas para el desarrollo de soluciones de aprendizaje automático en la vida real.  