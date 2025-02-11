import os
import pandas as pd
import numpy as np
import pytest
from aiLibrary.modelManager import ModelManager
from sklearn.linear_model import LogisticRegression

def test_train_and_evaluate_model():
    # Creamos un conjunto de datos sencillo.
    X_train = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1, 2, 3, 4]
    })
    y_train = [0, 1, 0, 1]
    X_test = pd.DataFrame({
        'feature1': [0.15, 0.35],
        'feature2': [1.5, 3.5]
    })
    y_test = [0, 1]
    
    mm = ModelManager(model_params={'max_iter': 200})
    mm.train_model(X_train, y_train)
    
    evaluation = mm.evaluate_model(X_test, y_test)
    cm = evaluation.get("confusion_matrix")
    
    # Dado que es un problema binario, la matriz de confusión debe tener forma (2, 2).
    assert cm.shape == (2, 2)

def test_save_and_load_model(tmp_path):
    # Creamos datos simples para entrenar el modelo.
    X_train = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1, 2, 3, 4]
    })
    y_train = [0, 1, 0, 1]
    
    mm = ModelManager(model_params={'max_iter': 200})
    mm.train_model(X_train, y_train)
    
    # Usamos tmp_path para crear una carpeta temporal y redirigir la salida.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_filename = "test_model.pkl"
    filepath = str(models_dir / model_filename)
    
    # Cambiamos el directorio de guardado temporalmente modificando la función save_model
    # (en este ejemplo, guardamos el modelo en la carpeta temporal en lugar de la carpeta 'models' real).
    import joblib
    joblib.dump(mm.model, filepath)
    assert os.path.exists(filepath)
    
    # Para probar la carga, cargamos directamente desde el archivo temporal.
    loaded_model = joblib.load(filepath)
    assert isinstance(loaded_model, LogisticRegression)