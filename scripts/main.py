from load_data import cargar_datos
from train import entrenar_modelo
from predict import predecir_disponibilidad

def main():
    # Cargar los datos
    datos = cargar_datos()
    
    # Entrenar el modelo con los datos cargados
    modelo = entrenar_modelo(datos)
    
    # Realizar predicciones de disponibilidad
    predicciones = predecir_disponibilidad(modelo, datos)
    
    # Mostrar las predicciones en consola
    print(predicciones)

if __name__ == '__main__':
    main()
