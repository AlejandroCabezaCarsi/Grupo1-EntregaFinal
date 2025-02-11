import os
import tempfile
import pandas as pd
import numpy as np
import pytest
import joblib

from aiLibrary.dataProcessor import DataProcessor
from aiLibrary.modelManager import ModelManager
from aiLibrary.predictionEngine import PredictionEngine

def test_full_workflow_integration():
    """
    Test de integración que simula el flujo completo:
    - Cargar y preprocesar un dataset simulado.
    - Dividir en conjuntos de entrenamiento y prueba.
    - Entrenar y evaluar el modelo.
    - Guardar y cargar el modelo.
    - Realizar predicciones y verificar la consistencia de la salida.
    """
    # Crear un dataset simulado (similar a heart.csv pero reducido)
    data = pd.DataFrame({
        'age': [52, 53, 70, 61, 62, 58],
        'sex': [1, 1, 1, 1, 0, 0],
        'cp': [0, 0, 0, 0, 0, 0],
        'trestbps': [125, 140, 145, 148, 138, 100],
        'chol': [212, 203, 174, 203, 294, 248],
        'fbs': [0, 1, 0, 0, 1, 0],
        'restecg': [1, 0, 1, 1, 1, 0],
        'thalach': [168, 155, 125, 161, 106, 122],
        'exang': [0, 1, 1, 0, 0, 0],
        'oldpeak': [1, 3.1, 2.6, 0, 1.9, 1],
        'slope': [2, 0, 0, 2, 1, 1],
        'ca': [2, 0, 0, 1, 3, 0],
        'thal': [3, 3, 3, 3, 2, 2],
        'target': [0, 0, 0, 0, 0, 1]  # datos de ejemplo (pueden ser ajustados)
    })

    # Preprocesamiento
    dp = DataProcessor(scaling_method='minmax')
    data_clean = dp.clean_data(data)
    data_transformed = dp.transform_categorical(data_clean)
    data_scaled = dp.scale_data(data_transformed, target_column='target')
    X_train, X_test, y_train, y_test = dp.split_data(data_scaled, target='target', test_size=0.5, random_state=42)

    # Entrenamiento y evaluación del modelo
    mm = ModelManager(model_params={'max_iter': 100})
    model = mm.train_model(X_train, y_train)
    eval_dict = mm.evaluate_model(X_test, y_test)
    cm = eval_dict.get("confusion_matrix")
    # Verificamos que la matriz de confusión tenga la forma (2, 2)
    assert cm.shape == (2, 2)

    # Guardar y cargar el modelo en un directorio temporal
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_filename = "temp_model.pkl"
        filepath = os.path.join(tmpdirname, model_filename)
        joblib.dump(model, filepath)
        loaded_model = joblib.load(filepath)
        
        # Verificar que el modelo cargado es del mismo tipo
        assert isinstance(loaded_model, type(model))
        
        # Predicción con el modelo cargado
        pe = PredictionEngine(model=loaded_model, scaler=dp.scaler, target_column='target')
        predictions = pe.evaluate_new_dataframe(data_scaled)
        # Verificamos que el número de predicciones coincide con el número de filas del dataset
        assert len(predictions) == len(data_scaled)

def test_error_on_empty_dataframe():
    """
    Verifica que se lance una excepción o se gestione correctamente
    el caso en el que se suministra un DataFrame vacío al preprocesamiento.
    """
    dp = DataProcessor()
    empty_df = pd.DataFrame()
    
    # Dependiendo de la implementación, split_data sobre un DataFrame vacío debería lanzar un error.
    with pytest.raises(KeyError):
        dp.split_data(empty_df, target='target')

def test_save_model_without_training(tmp_path):
    """
    Verifica que se genere un error si se intenta guardar el modelo sin haberlo entrenado.
    Para esto, se crea una instancia de ModelManager sin llamar a train_model.
    """
    mm = ModelManager()
    # Si el modelo no está entrenado, podríamos esperar que la función save_model lance un error.
    # En la implementación actual, save_model depende de que mm.model haya sido ajustado.
    with pytest.raises(Exception):
        # Intentamos guardar el modelo sin entrenamiento.
        mm.save_model("untrained_model.pkl")

def test_prediction_consistency_after_reload():
    """
    Verifica que las predicciones sean consistentes antes y después de guardar y recargar el modelo.
    """
    # Crear un pequeño dataset
    X = pd.DataFrame({
        'feature1': [0.2, 0.4, 0.6, 0.8],
        'feature2': [1, 2, 3, 4]
    })
    y = [0, 0, 1, 1]
    
    # Entrenar el modelo
    mm = ModelManager(model_params={'max_iter': 100})
    mm.train_model(X, y)
    preds_before = mm.model.predict(X)
    
    # Guardar el modelo en un archivo temporal
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_filename = "consistency_model.pkl"
        filepath = os.path.join(tmpdirname, model_filename)
        joblib.dump(mm.model, filepath)
        loaded_model = joblib.load(filepath)
        preds_after = loaded_model.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)
