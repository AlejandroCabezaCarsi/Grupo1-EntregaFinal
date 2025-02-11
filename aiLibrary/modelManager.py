import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class ModelManager:
    def __init__(self, model_params: dict = None):
        """
        Inicializa el ModelManager con los parámetros del modelo.
        
        :param model_params: Diccionario con parámetros para LogisticRegression.
        """
        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.model = LogisticRegression(**model_params)

    def train_model(self, X_train, y_train):
        """
        Entrena el modelo utilizando el conjunto de entrenamiento.
        
        :param X_train: Características de entrenamiento.
        :param y_train: Etiquetas de entrenamiento.
        :return: El modelo entrenado.
        """
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self, X_test, y_test) -> dict:
        """
        Evalúa el modelo y genera la matriz de confusión.
        
        :param X_test: Características del conjunto de prueba.
        :param y_test: Etiquetas del conjunto de prueba.
        :return: Diccionario con la matriz de confusión.
        """
        predictions = self.model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        return {"confusion_matrix": cm}

    def save_model(self, model_name: str) -> str:
        """
        Guarda el modelo entrenado en la carpeta 'models'.
        
        :param model_name: Nombre del archivo donde se guardará el modelo (por ejemplo, 'model.pkl').
        :return: Ruta del archivo donde se guardó el modelo.
        """
        directory = 'models'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, model_name)
        joblib.dump(self.model, filepath)
        return filepath

    def load_model(self, model_name: str):
        """
        Carga un modelo previamente guardado.
        
        :param model_name: Nombre del archivo del modelo a cargar.
        :return: El modelo cargado.
        """
        directory = 'models'
        filepath = os.path.join(directory, model_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modelo no encontrado en {filepath}")
        self.model = joblib.load(filepath)
        return self.model
