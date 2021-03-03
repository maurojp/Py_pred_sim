import pandas as pd
import numpy as np
import threading
import time
import queue
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

# Lista para guardar las predicciones
history = []

# Parametros de normalización para conjunto TS1-4_COOLER
T_C_0_num = 30.355
T_C_0_den = 31.739

# Función de normalización
def normalize(x):
  return (x - T_C_0_num) / T_C_0_den

# Logica del proceso Simulador
def simulator(thread_name, console_lock):
    for i in range(len(sds)):
        # Pausa de un segundo entre entregas de datos
        time.sleep(1)
        # Inserción de los datos en el repositorio FIFO
        q.put(sds[i])
        with console_lock:
            print("Proceso Simulador de Lecturas - Proc.Nro. " + str(threading.get_ident()))
            print("Lecturas enviadas: ", sds[i])


# Logica del proceso predictor
def predictor(thread_name, console_lock, loaded_model):
    pred_counter = 0
    serie = np.empty((1, 4))
    while (pred_counter < 150):
        if q.qsize() > 0: # Hay datos en el repositorio
            b = q.get()
            with console_lock:
                print("Proceso Predictor - Proc.Nro. " + str(threading.get_ident()))
                print("Prediccion Nro.: ", pred_counter, " - Datos Recibidos: ", b)
                print("Objetos restantes en la cola: " + str(q.qsize()))
                # Preparado de la Serie
                if pred_counter == 0:
                    serie = np.array([b])
                else:
                    serie = np.append(serie, np.array([b]), axis=0)
                # Grafica de la serie acumulada
                plt.plot(serie)
                # Plot de la salida normalizada a la cota máxima de temperaturas para mejor visualización
                plt.plot([h * 30 for h in history])
                plt.show()

                # Predicción
                if len(serie) > 9: # Hay suficientes datos para predecir
                    x = normalize(serie[-10:, :])
                    # print("Ultimas 10 lecturas recibidas")
                    # print(x)
                    pred_val = loaded_model.predict(np.array([x]))
                    i_pred_val = np.argmax(pred_val)
                    history.append(i_pred_val)
                    print("Predicción: ", pred_val, "Etiqueta: ", i_pred_val)
                    print("Estado en 50 segundos: ", text_labels[i_pred_val])

                else:
                    print("No hay suficientes datos para realizar una predicción.")
                    history.append(2)
                pred_counter += 1
        else:
            time.sleep(1)
    # Guardo en archivo el resultado de las predicciones
    mat = np.matrix(history)
    with open('resultados_h_p_s_1.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.4f', delimiter='\t')

# Lectura del dataset sintetico
sds = pd.read_csv('SyntheticTimeSerie_1.txt', sep='\t', header=None)
sds = sds.to_numpy()

# Semaforo para acceso a la consola
console_lock = threading.Lock()

# Lectura del modelo de aprendizaje profundo
with open('model_m.json', 'r') as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)

# Lectura de los parámetros entrenados
loaded_model.load_weights('model_m.h5')

# Compilación del modelo
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Etiquetas con los estados posibles de salida de predicción
text_labels = ['ESTADO CRITICO', 'ESTADO MEDIO', 'EFICIENCIA TOTAL']

# Creación del repositorio para intercambio de datos entre procesos (FIFO)
q = queue.Queue()

# Visualización de la estructura del modelo de aprendizaje profundo leido
loaded_model.summary()

# Creación de los procesos concurrentes
p1 = threading.Thread(target=simulator, args=('p1', console_lock))
p2 = threading.Thread(target=predictor, args=('p2', console_lock, loaded_model))

# Inicio de la ejecución de los procesos
p1.start()
p2.start()
