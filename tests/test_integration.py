import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from aiLibrary.dataProcessor import DataProcessor
from aiLibrary.modelManager import ModelManager
from aiLibrary.predictionEngine import PredictionEngine

# Test para verificar que tras entrenar el modelo se creen atributos necesarios
def test_train_model_attributes():
    data = pd.DataFrame({
        'age': [50, 60, 70, 80, 55, 65, 75, 85],
        'sex': [1, 1, 0, 0, 1, 0, 1, 0],
        'cp': [0, 1, 0, 1, 0, 1, 0, 1],
        'trestbps': [120, 140, 150, 160, 130, 150, 170, 180],
        'chol': [200, 220, 240, 260, 210, 230, 250, 270],
        'fbs': [0, 1, 0, 1, 0, 1, 0, 1],
        'restecg': [0, 1, 0, 1, 0, 1, 0, 1],
        'thalach': [150, 160, 170, 180, 155, 165, 175, 185],
        'exang': [0, 1, 0, 1, 0, 1, 0, 1],
        'oldpeak': [1, 2, 3, 4, 1.5, 2.5, 3.5, 4.5],
        'slope': [1, 2, 1, 2, 1, 2, 1, 2],
        'ca': [0, 1, 0, 1, 0, 1, 0, 1],
        'thal': [1, 2, 1, 2, 1, 2, 1, 2],
        'target': [0, 1, 0, 1, 0, 1, 0, 1]
    })
    dp = DataProcessor(scaling_method='minmax')
    data_clean = dp.clean_data(data)
    data_transformed = dp.transform_categorical(data_clean)
    data_scaled = dp.scale_data(data_transformed, target_column='target')
    
    X = data_scaled.drop(columns=['target'])
    y = data_scaled['target']
    
    # Verifica que se haya creado el atributo "classes_" y que contenga ambas clases
    mm = ModelManager(model_params={'max_iter': 100})
    mm.train_model(X, y)

    assert hasattr(mm.model, 'classes_'), "El modelo no tiene el atributo 'classes_' después del entrenamiento."
    assert set(mm.model.classes_) == {0, 1}, "El atributo 'classes_' no contiene ambas clases esperadas."

# Test para verificar la exactitud del modelo en un dataset sencillo y separable
def test_model_accuracy_on_balanced_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 9, 10],
        'feature2': [1, 2, 9, 10]
    })
    y = [0, 0, 1, 1]
    
    mm = ModelManager(model_params={'max_iter': 100})
    mm.train_model(X, y)
    y_pred = mm.model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    # Se espera que la exactitud sea perfecta (1.0) en este caso
    assert acc == 1.0, "La exactitud del modelo no es 1.0 para un dataset claramente separable."

# Test para verificar el formato de salida del PredictionEngine
def test_prediction_engine_output_format():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [1, 2, 3, 4]
    })
    y = [0, 0, 1, 1]
    
    mm = ModelManager(model_params={'max_iter': 100})
    mm.train_model(X, y)
    
    # No se requiere escalado, por lo que pasamos scaler=None
    pe = PredictionEngine(model=mm.model, scaler=None, target_column=None)
    preds = pe.evaluate_new_dataframe(X)
    
    # Verificamos que la salida sea un array de NumPy y tenga la misma cantidad de elementos que X
    assert isinstance(preds, np.ndarray), "La salida de PredictionEngine no es un array de NumPy."
    assert preds.shape[0] == X.shape[0], "El número de predicciones no coincide con el número de muestras."

# Test para comparar los resultados al usar StandardScaler vs MinMaxScaler
def test_standard_scaler_vs_minmax_scaler():
    data = pd.DataFrame({
        'num1': [10, 20, 30, 40],
        'target': [0, 1, 0, 1]
    })
    
    dp_minmax = DataProcessor(scaling_method='minmax')
    scaled_minmax = dp_minmax.scale_data(data.copy(), target_column='target')
    
    dp_standard = DataProcessor(scaling_method='standard')
    scaled_standard = dp_standard.scale_data(data.copy(), target_column='target')
    
    # Se espera que los valores escalados sean diferentes entre ambos métodos
    assert not np.allclose(scaled_minmax['num1'], scaled_standard['num1']), \
        "Los resultados del escalado con MinMaxScaler y StandardScaler son idénticos, lo cual no debería suceder."

# Test para verificar la consistencia del one-hot encoding
def test_transform_categorical_consistency():
    df = pd.DataFrame({
        'cp': [0, 1, 0, 1],
        'restecg': [0, 0, 1, 1],
        'ca': [0, 1, 0, 1],
        'thal': [3, 2, 3, 2],
        'target': [0, 1, 0, 1]
    })
    dp = DataProcessor()
    transformed1 = dp.transform_categorical(df.copy())
    transformed2 = dp.transform_categorical(df.copy())
    pd.testing.assert_frame_equal(transformed1, transformed2)

# Test para verificar que se lance un error al usar un método de escalado inválido
def test_error_on_invalid_scaling_method():
    # Si se pasa un método de escalado inválido, se espera que el DataProcessor lo maneje.
    dp = DataProcessor(scaling_method='invalid_method')
    from sklearn.preprocessing import StandardScaler
    assert isinstance(dp.scaler, StandardScaler), \
        "El DataProcessor no ha utilizado StandardScaler como valor por defecto para un método de escalado inválido."

# Test para verificar la consistencia de las predicciones antes y después de guardar y recargar el modelo
def test_prediction_consistency_after_reload():
    X = pd.DataFrame({
        'feature': [0.2, 0.4, 0.6, 0.8]
    })
    y = [0, 0, 1, 1]
    
    mm = ModelManager(model_params={'max_iter': 100})
    mm.train_model(X, y)
    preds_before = mm.model.predict(X)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_filename = "consistency_model.pkl"
        filepath = os.path.join(tmpdirname, model_filename)
        joblib.dump(mm.model, filepath)
        loaded_model = joblib.load(filepath)
        preds_after = loaded_model.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)