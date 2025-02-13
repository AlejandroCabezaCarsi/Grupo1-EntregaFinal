import pandas as pd
from aiLibrary.dataProcessor import DataProcessor
from aiLibrary.modelManager import ModelManager
from aiLibrary.predictionEngine import PredictionEngine

def main():
    # Definir el nombre del archivo CSV y la columna objetivo
    csv_file = "heart.csv"
    target_column = "target"

    # Instanciar DataProcessor con el método de escalado deseado ('minmax' o 'standard')
    dp = DataProcessor(scaling_method='minmax')
    
    # Cargar el dataset
    data = dp.load_data(csv_file)
    
    # Limpiar los datos (eliminar filas con datos faltantes)
    data = dp.clean_data(data)
    
    # Transformar las variables categóricas (One-Hot Encoding)
    data = dp.transform_categorical(data)
    
    # Escalar las variables numéricas (se omite la columna objetivo)
    data = dp.scale_data(data, target_column=target_column)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = dp.split_data(data, target=target_column)
    
    # Definir parámetros para el modelo de regresión logística
    model_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 100
    }
    
    # Instanciar y entrenar el modelo con ModelManager
    mm = ModelManager(model_params=model_params)
    model = mm.train_model(X_train, y_train)
    
    # Evaluar el modelo y mostrar la matriz de confusión
    evaluation = mm.evaluate_model(X_test, y_test)
    print("Matriz de Confusión:")
    print(evaluation["confusion_matrix"])
    
    # Guardar el modelo entrenado
    model_path = mm.save_model("heart_disease_model.pkl")
    print("Modelo guardado en:", model_path)
    
    # Cargar el modelo guardado (para demostrar la funcionalidad)
    loaded_model = mm.load_model("heart_disease_model.pkl")
    print("Modelo cargado correctamente.")
    
    # Realizar predicciones utilizando el PredictionEngine
    # Se utiliza el mismo dataset para demostración; en un caso real se emplearían nuevos datos.
    pe = PredictionEngine(model=loaded_model, scaler=dp.scaler, target_column=target_column)
    predictions = pe.evaluate_new_dataframe(data)
    print("Predicciones sobre el dataset:")
    print([int(x) for x in predictions])

if __name__ == "__main__":
    main()
