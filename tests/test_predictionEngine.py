import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from aiLibrary.predictionEngine import PredictionEngine

def test_prediction_engine():
    # Creamos un conjunto de datos simple.
    data = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1, 2, 3, 4],
        'target': [0, 1, 0, 1]
    })
    
    # Separamos las características y la etiqueta.
    X = data[['feature1', 'feature2']]
    y = data['target']
    
    # Entrenamos un modelo simple.
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    
    # Ajustamos un escalador en el conjunto de características.
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    # Creamos un nuevo DataFrame para predecir, incluyendo la columna 'target' para comprobar que se elimina.
    new_data = pd.DataFrame({
        'feature1': [0.15, 0.35],
        'feature2': [1.5, 3.5],
        'target': [0, 1]
    })
    
    # Inicializamos el PredictionEngine.
    pe = PredictionEngine(model=model, scaler=scaler, target_column='target')
    predictions = pe.evaluate_new_dataframe(new_data)
    
    # Verificamos que el número de predicciones coincide con el número de muestras en new_data.
    assert len(predictions) == len(new_data)