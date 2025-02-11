# Diagnóstico Asistido de Enfermedades Cardiovasculares

Este proyecto es una librería de alto nivel en Python diseñada para integrar técnicas de inteligencia artificial en aplicaciones de diagnóstico médico. La herramienta se centra en la clasificación de enfermedades cardiovasculares utilizando un modelo de regresión logística y ofrece funcionalidades para:

- **Preprocesamiento de datos:** carga, limpieza, transformación (One-Hot Encoding) y escalado.
- **Entrenamiento y evaluación:** entrenamiento de un modelo de regresión logística y evaluación mediante una matriz de confusión.
- **Gestión del modelo:** guardado y carga del modelo entrenado.
- **Predicción:** generación de predicciones sobre nuevos datos aplicando el mismo preprocesamiento.

---

## Estructura del Proyecto

- **aiLibrary/**: Directorio que contiene el código fuente de la librería.
  - ****init**.py**: Archivo que inicializa el paquete e importa las clases principales.
  - **dataProcessor.py**: Módulo que maneja el preprocesamiento de datos, incluyendo carga, limpieza, transformación y escalado.
  - **modelManager.py**: Módulo responsable del entrenamiento, evaluación y gestión del modelo de regresión logística.
  - **predictionEngine.py**: Módulo que genera predicciones sobre nuevos datos aplicando el mismo preprocesamiento utilizado durante el entrenamiento.

- **tests/**: Directorio que contiene los scripts de pruebas unitarias, utilizando pytest.
  - **test_dataProcessor.py**: Pruebas unitarias para el módulo `dataProcessor`.
  - **test_modelManager.py**: Pruebas unitarias para el módulo `modelManager`.
  - **test_predictionEngine.py**: Pruebas unitarias para el módulo `predictionEngine`.

- **App.py**: Script de aplicación de ejemplo que demuestra el flujo completo de preprocesamiento, entrenamiento, evaluación y predicción.

- **heart.csv**: Dataset de ejemplo que contiene datos de pacientes y etiquetas para la clasificación de enfermedades cardiovasculares.

- **requirements.txt**: Archivo que lista todas las dependencias necesarias para ejecutar el proyecto.

---

## Requisitos

- **Python:** Versión 3.13 (u otra versión compatible).
- **Dependencias:** Las librerías necesarias se encuentran listadas en el archivo `requirements.txt`. Entre ellas se incluyen:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `seaborn`

---

## Instalación (Windows)

Sigue estos pasos para configurar el entorno y preparar el proyecto:

1. **Clonar el repositorio:**

   Abre una terminal y ejecuta:

   ```bash
   git clone https://github.com/tu-usuario/Grupo1-EntregaFinal.git
   cd Grupo1-EntregaFinal
   ```

2. **Crear un entorno virtual:**

   Crea un entorno virtual con Python 3.13:

   ```bash
   python3 -m venv venv
   ```

   Activa el entorno virtual:

   ```bash
   .venv\Scripts\activate
    ```

3. **Instalar dependencias:**

   Instala las dependencias necesarias:

   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. **Aplicacion**

    Para ejecutar la aplicación de ejemplo, utiliza el siguiente comando:

    ```bash
    python App.py
    ```

2. **Pruebas**

    Para ejecutar las pruebas unitarias, utiliza el siguiente comando:

    ```bash
    pytest
    ```
