import pandas as pd
import numpy as np
import os
import tempfile
import pytest
from aiLibrary.dataProcessor import DataProcessor

def test_clean_data():
    # Creamos un DataFrame con algunos valores faltantes.
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 2, 3, 4],
        'target': [0, 1, 0, 1]
    })
    dp = DataProcessor(scaling_method='minmax')
    cleaned_df = dp.clean_data(df)
    
    # Verificamos que no queden valores nulos.
    assert not cleaned_df.isnull().values.any()
    # Dado que solo la segunda y cuarta fila están completas, se esperan 2 filas.
    assert len(cleaned_df) == 2

def test_transform_categorical():
    # Creamos un DataFrame con las columnas categóricas esperadas.
    df = pd.DataFrame({
        'cp': [0, 1, 2],
        'restecg': [0, 1, 2],
        'ca': [0, 1, 2],
        'thal': [0, 1, 2],
        'target': [0, 1, 0]
    })
    dp = DataProcessor()
    transformed_df = dp.transform_categorical(df)
    
    # Verificamos que las columnas originales no existan después de aplicar one-hot encoding.
    for col in ['cp', 'restecg', 'ca', 'thal']:
        assert col not in transformed_df.columns

def test_scale_data():
    # Creamos un DataFrame simple con columnas numéricas.
    df = pd.DataFrame({
        'age': [50, 60, 70],
        'trestbps': [120, 140, 160],
        'target': [0, 1, 0]
    })
    dp = DataProcessor(scaling_method='minmax')
    scaled_df = dp.scale_data(df.copy(), target_column='target')
    
    # Verificamos que las columnas numéricas se hayan escalado entre 0 y 1.
    for col in ['age', 'trestbps']:
        assert scaled_df[col].min() == 0.0
        assert scaled_df[col].max() == 1.0

def test_split_data():
    # Creamos un DataFrame simple.
    df = pd.DataFrame({
        'age': [50, 60, 70, 80],
        'trestbps': [120, 140, 160, 180],
        'target': [0, 1, 0, 1]
    })
    dp = DataProcessor()
    X_train, X_test, y_train, y_test = dp.split_data(df, target='target', test_size=0.5, random_state=42)
    
    # Con test_size del 50% y 4 muestras, se esperan 2 muestras en cada conjunto.
    assert len(X_train) == 2
    assert len(X_test) == 2
