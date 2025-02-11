import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataProcessor:
    def __init__(self, scaling_method='minmax'):
        """
        Inicializa el DataProcessor con el método de escalado elegido.
        
        :param scaling_method: 'minmax' para MinMaxScaler o cualquier otro valor para StandardScaler.
        """
        self.scaling_method = scaling_method
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carga un archivo CSV en un DataFrame de Pandas.
        
        :param filepath: Ruta del archivo CSV.
        :return: DataFrame con los datos.
        """
        return pd.read_csv(filepath)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina filas con datos faltantes.
        
        :param data: DataFrame original.
        :return: DataFrame limpio.
        """
        return data.dropna()

    def transform_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica One-Hot Encoding a las columnas categóricas 'cp', 'restecg', 'ca' y 'thal'.
        
        :param data: DataFrame a transformar.
        :return: DataFrame con las variables categóricas transformadas.
        """
        categorical_cols = ['cp', 'restecg', 'ca', 'thal']
        for col in categorical_cols:
            if col in data.columns:
                data = pd.get_dummies(data, columns=[col], prefix=col)
        return data

    def scale_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Escala las columnas numéricas (excepto la columna objetivo) utilizando el escalador elegido.
        
        :param data: DataFrame con los datos.
        :param target_column: Nombre de la columna objetivo que no se debe escalar.
        :return: DataFrame con las columnas numéricas escaladas.
        """
        numeric_cols = [col for col in data.columns 
                        if col != target_column and data[col].dtype in [np.int64, np.float64]]
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        return data

    def split_data(self, data: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Divide el DataFrame en conjuntos de entrenamiento y prueba.
        
        :param data: DataFrame preprocesado.
        :param target: Nombre de la columna objetivo.
        :param test_size: Proporción de datos para el conjunto de prueba.
        :param random_state: Semilla para la división.
        :return: Tuple (X_train, X_test, y_train, y_test).
        """
        X = data.drop(columns=[target])
        y = data[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
