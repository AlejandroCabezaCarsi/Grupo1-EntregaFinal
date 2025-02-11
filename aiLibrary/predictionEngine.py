import pandas as pd
import numpy as np

class PredictionEngine:
    def __init__(self, model, scaler, target_column: str = None):
        """
        Inicializa el PredictionEngine.
        
        :param model: Modelo entrenado.
        :param scaler: Escalador ya ajustado durante el preprocesamiento.
        :param target_column: Nombre de la columna objetivo (se elimina si está presente).
        """
        self.model = model
        self.scaler = scaler
        self.target_column = target_column

    def evaluate_new_dataframe(self, data: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones en nuevos datos.
        
        :param data: DataFrame con nuevos datos.
        :return: Numpy array con las predicciones.
        """
        # Eliminar filas con datos faltantes
        data = data.dropna()

        # Transformar variables categóricas mediante One-Hot Encoding
        categorical_cols = ['cp', 'restecg', 'ca', 'thal']
        for col in categorical_cols:
            if col in data.columns:
                data = pd.get_dummies(data, columns=[col], prefix=col)

        # Si existe la columna objetivo, eliminarla
        if self.target_column and self.target_column in data.columns:
            data = data.drop(columns=[self.target_column])

        # Escalar columnas numéricas usando el escalador ajustado
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if self.scaler is not None:
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])

        predictions = self.model.predict(data)
        return predictions
