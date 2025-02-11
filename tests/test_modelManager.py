import os
import pandas as pd
import pytest
from aiLibrary.modelManager import ModelManager
from sklearn.linear_model import LogisticRegression
import joblib

# Test para verificar que el modelo se entrena y se evalúa correctamente.
def test_train_and_evaluate_model():
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
    
    # Inicializamos ModelManager con parámetros para LogisticRegression.
    mm = ModelManager(model_params={'max_iter': 200})
    # Entrenamos el modelo con el conjunto de entrenamiento.
    mm.train_model(X_train, y_train)
    
    # Evaluamos el modelo sobre el conjunto de prueba y obtenemos la matriz de confusión.
    evaluation = mm.evaluate_model(X_test, y_test)
    cm = evaluation.get("confusion_matrix")
    
    # Comprobamos que la matriz de confusión tiene forma (2, 2) (para un problema binario).
    assert cm.shape == (2, 2)

# Test para verificar que guardar y cargar el modelo funciona correctamente.
def test_save_and_load_model(tmp_path):
    X_train = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1, 2, 3, 4]
    })
    y_train = [0, 1, 0, 1]
    
    # Inicializamos ModelManager y entrenamos el modelo.
    mm = ModelManager(model_params={'max_iter': 200})
    mm.train_model(X_train, y_train)
    
    # Creamos un directorio temporal para guardar el modelo.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_filename = "test_model.pkl"
    
    # Guardamos el modelo. El método save_model crea la carpeta "models" en la raíz, pero para la prueba se verifica que el archivo guardado existe.
    saved_path = mm.save_model(model_filename)
    assert os.path.exists(saved_path), "El archivo del modelo guardado no existe."
    
    # Cargamos el modelo y verificamos que es una instancia de LogisticRegression.
    loaded_model = mm.load_model(model_filename)
    assert isinstance(loaded_model, LogisticRegression), "El modelo cargado no es de tipo LogisticRegression."

# Test para verificar que intentar guardar un modelo sin entrenarlo lanza un ValueError.
def test_save_model_without_training(tmp_path):
    # Inicializamos ModelManager sin entrenar el modelo.
    mm = ModelManager()
    # Se espera que, al intentar guardar el modelo sin entrenarlo, se lance un ValueError.
    with pytest.raises(ValueError):
        mm.save_model("untrained_model.pkl")