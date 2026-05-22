# lab---5
 ## Parte A
 Se realizó el fundamento teórico necesario para comprender la práctica, donde se investigaron y explicaron conceptos esenciales relacionados con la actividad simpática y parasimpática del sistema nervioso autónomo, su efecto sobre la frecuencia cardíaca y la variabilidad de la frecuencia cardíaca (HRV) obtenida a partir de la señal ECG, así como el diagrama de Poincaré como herramienta de análisis de la serie R-R. Con base en esto, se formuló un plan de acción representado en un diagrama de flujo y se realizó la adquisición de la señal ECG de un sujeto de prueba durante cuatro minuto.
 ### A.
 ### 1. Actividad simpática y parasimpática del sistema nervioso autónomo
El sistema nervioso autónomo (SNA) regula funciones involuntarias del cuerpo como la frecuencia cardíaca, la presión arterial y la digestión. Se divide en dos ramas principales que actúan de forma opuesta y complementaria. La rama parasimpática (o vagal) funciona como un "freno" sobre el corazón, disminuyendo la frecuencia cardíaca mediante la liberación de acetilcolina. La rama simpática, por su parte, actúa como un "acelerador", aumentando la frecuencia cardíaca especialmente en situaciones de estrés o cambios posturales.

### 2. Efecto de la actividad simpática y parasimpática en la frecuencia cardíaca
La frecuencia cardíaca está directamente modulada por el balance entre ambas ramas del sistema autónomo. Cuando predomina la actividad simpática, el corazón late más rápido debido al aumento en la velocidad de despolarización del nodo Sinusal, reduciendo la variabilidad y haciendo el ritmo más rígido y constante. En cambio, cuando predomina la actividad parasimpática, el nervio vago frena la frecuencia cardíaca y aumenta la variabilidad entre latidos, por lo que una mayor influencia vagal se asocia a una HRV más alta. Los resultados de Toichi et al. confirmaron este principio: el bloqueo parasimpático con atropina disminuyó significativamente el índice vagal cardíaco (CVI) en todas las condiciones posturales, mientras que el bloqueo simpático con propranolol redujo el índice simpático cardíaco (CSI) de forma más prominente en posición de pie, condición en la que la actividad simpática es fisiológicamente mayor, lo cual es coherente con el predominio parasimpático durante el reposo en posición supina.

### 3. Variabilidad de la frecuencia cardíaca (HRV) obtenida a partir de la señal electrocardiográfica (ECG)
La HRV se calcula analizando los intervalos R-R extraídos de la señal de ECG, es decir, el tiempo que separa cada latido consecutivo. Estos intervalos cambian de manera natural debido a la modulación autonómica sobre el nodo Sinusal. Una alta variabilidad suele indicar un sistema parasimpático activo, flexible y capaz de adaptarse, mientras que una variabilidad reducida puede asociarse a estrés, fatiga, exigencia cognitiva o predominio simpático. En el estudio de Toichi et al., la señal ECG se registró desde la derivación estándar II, se digitalizó a 2000 Hz con una precisión de 1 ms, y se analizaron 3 minutos consecutivos de intervalos R-R sin artefactos. Los mismos datos fueron procesados con tres métodos distintos: el diagrama de Lorenz/Poincaré, el análisis espectral mediante FFT y el coeficiente de variación (CV), siendo el diagrama de Poincaré el que arrojó los índices más confiables y consistentes.

### 4. Diagrama de Poincaré como herramienta de análisis de la serie R-R. 
El diagrama de Poincaré es una técnica de análisis no lineal que representa cada intervalo R-R en función del intervalo inmediatamente siguiente, generando una nube de puntos cuya forma geométrica informa sobre la dinámica autonómica del corazón. Cuando la dispersión es amplia, la variabilidad es alta y predomina la influencia vagal; cuando la nube adopta una forma elongada y estrecha, la variabilidad es menor y el tono simpático es dominante. Toichi et al. aplicaron este método extrayendo dos medidas del eje de la elipse resultante: el eje longitudinal L, paralelo a la línea de identidad, que refleja la amplitud global de la fluctuación, y el eje transverso T, perpendicular a dicha línea, que captura la variación latido a latido. . A partir de estas medidas construyeron el índice vagal cardíaco (CVI = log₁₀(L × T)) y el índice simpático cardíaco (CSI = L/T), los cuales demostraron ser más robustos que los índices obtenidos por espectroscopía o coeficiente de variación. Adicionalmente, este método tiene la ventaja práctica de requerir menos de 2 minutos de registro y no exigir control de la respiración.

### 5.Variabilidad de la frecuencia cardíaca (HRV) y balance autonómico 
La HRV constituye una ventana no invasiva hacia el funcionamiento del sistema nervioso autónomo, ya que las fluctuaciones en los intervalos R-R son consecuencia directa de la modulación simultánea ejercida por las ramas simpática y parasimpática sobre el corazón. Un valor elevado de HRV refleja predominio vagal y una buena capacidad de adaptación fisiológica ante distintas demandas del entorno, mientras que una HRV disminuida se interpreta como señal de mayor influencia simpática, reducción en la flexibilidad regulatoria o presencia de estados de estrés o enfermedad. Para cuantificar este balance se emplean distintos enfoques analíticos: el dominio temporal, el dominio frecuencial y métodos no lineales como el diagrama de Poincaré. En el trabajo de Toichi et al., este último enfoque permitió caracterizar de forma simultánea e independiente el tono vagal y simpático mediante los índices CVI y CSI, los cuales respondieron de manera consistente a los bloqueos farmacológicos en todas las condiciones evaluadas, demostrando mayor fiabilidad que los métodos convencionales para describir el balance autonómico en condiciones experimentales y clínicas.

<img width="281" height="626" alt="image" src="https://github.com/user-attachments/assets/be375fd6-0d56-45e3-a50b-050a2d9c228a" />

### B.
**Código de adquisión de la señal con filtro IIR**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nidaqmx
from nidaqmx.constants import AcquisitionType
from threading import Thread, Event
from collections import deque
import datetime
import time
from scipy import signal  

fs = 2000        
canal = "Dev6/ai0"    
tamano_bloque = int(fs * 0.05)   
ventana_tiempo = 5.0             

# -----------------------------
# NUEVO FILTRO IIR PASA-BANDA
# H(s) = 51031 s / (s^2 + 51031 s + 2952.9)
# Convertimos a coeficientes discretos usando bilinear transform
# -----------------------------

b = [0, 51031, 0]           # Numerador: 51031*s --> [0, 51031, 0]
a = [1, 51031, 2952.9]      # Denominador: s^2 + 51031 s + 2952.9

# Convertimos a filtro digital con bilinear transform
b_d, a_d = signal.bilinear(b, a, fs=fs)

zi = signal.lfilter_zi(b_d, a_d)  # Condiciones iniciales

buffer_graf = deque(maxlen=int(fs * ventana_tiempo))
datos_guardados = []

adquiriendo = Event()
detener_hilo = Event()
thread_lectura = None

def hilo_lectura():
    global datos_guardados, buffer_graf, zi
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(canal)
    task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
    task.start()
    print(f"\n▶ Adquisición iniciada en {canal} ({fs} Hz).")

    while not detener_hilo.is_set():
        if adquiriendo.is_set():
            try:
                datos = np.array(task.read(number_of_samples_per_channel=tamano_bloque))
                
                # FILTRO PASA-BANDA IIR NUEVO
                datos_filtrados, zi = signal.lfilter(b_d, a_d, datos, zi=zi)
                
                buffer_graf.extend(datos_filtrados)
                datos_guardados.extend(datos_filtrados)

            except Exception as e:
                print("⚠ Error de lectura:", e)
                break
        else:
            time.sleep(0.05)

    task.stop()
    task.close()
    print("Adquisición detenida")

def iniciar(event):
    global thread_lectura
    if not adquiriendo.is_set():
        if thread_lectura is None or not thread_lectura.is_alive():
            detener_hilo.clear()
            thread_lectura = Thread(target=hilo_lectura, daemon=True)
            thread_lectura.start()
        adquiriendo.set()
        print("▶ Grabando...")

def detener(event):
    """Detiene y guarda los datos."""
    adquiriendo.clear()
    detener_hilo.set()
    time.sleep(0.3)

    if datos_guardados:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"ECG_filtrado__ahorasi_ultimaaaaa_otravezz{timestamp}.txt"
        tiempos = np.arange(len(datos_guardados)) / fs
        data = np.column_stack((tiempos, datos_guardados))
        np.savetxt(nombre_archivo, data, fmt="%.6f", header="Tiempo(s)\tVoltaje(V)")
        print(f"Señal guardada en {nombre_archivo} ({len(datos_guardados)} muestras)")
    else:
        print("No se capturaron datos")

# -----------------------------
# Configuración gráfica
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.25)
linea, = ax.plot([], [], lw=1.2, color='royalblue')
ax.set_xlim(0, ventana_tiempo)
ax.set_ylim(-2, 2)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Voltaje [V]")
ax.set_title("ECG filtrado (IIR tiempo real)")
ax.grid(True, linestyle="--", alpha=0.6)

x = np.linspace(0, ventana_tiempo, int(fs * ventana_tiempo))
y = np.zeros_like(x)

def actualizar(frame):
    if len(buffer_graf) > 0:
        y = np.array(buffer_graf)
        if len(y) < len(x):
            y = np.pad(y, (len(x)-len(y), 0), constant_values=0)
        linea.set_data(x, y)
    return linea,

ax_iniciar = plt.axes([0.3, 0.1, 0.15, 0.075])
ax_detener = plt.axes([0.55, 0.1, 0.2, 0.075])
btn_iniciar = Button(ax_iniciar, 'Iniciar', color='purple', hovercolor='purple')
btn_detener = Button(ax_detener, 'Detener y Guardar', color='pink', hovercolor='pink')
btn_iniciar.on_clicked(iniciar)
btn_detener.on_clicked(detener)

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, actualizar, interval=50, blit=True)
plt.tight_layout()
plt.show()
```
<br>
<img width="988" height="387" alt="image" src="https://github.com/user-attachments/assets/150d31cf-063d-4205-8e2f-f1a8c3bf3fd3" />

<br>

<img width="204" height="518" alt="image" src="https://github.com/user-attachments/assets/ff52d205-9f84-4ae6-b541-61a39b2071ad" />


## Parte B 
Se realizó el preprocesamiento de la señal ECG adquirida, diseñando e implementando un filtro digital IIR para eliminar el ruido y los artefactos presentes en el registro, del cual se obtuvo su ecuación en diferencias aplicándolo con condiciones iniciales en cero. Una vez filtrada, la señal fue segmentada en dos bloques de dos minutos, uno correspondiente al reposo y otro a la lectura en voz alta, en cada uno de los cuales se detectaron los picos R y se calcularon los intervalos R-R para obtener la serie temporal. A partir de esta información se compararon la media y la desviación estándar de los intervalos R-R entre ambas condiciones, permitiendo observar diferencias en la variabilidad de la frecuencia cardíaca asociadas al balance autonómico en cada estado.

*CALCULOS*
*ECUACION DEL FLTRO*



Señal capturada :

```
import pandas as pd
import matplotlib.pyplot as plt

archivo = "/content/ECG_tomado.txt"


df = pd.read_csv(
    archivo,
    delim_whitespace=True,
    skiprows=1,
    names=["Tiempo", "Voltaje"]
)


plt.figure(figsize=(15,5))
plt.plot(df["Tiempo"], df["Voltaje"], linewidth=1)

plt.title("ECG Filtrado")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.show()



```
División de la señal en segmentos de 2 minutos:
```


t = df["Tiempo(s)"]
ecg = df["Voltaje(V)"]

# Segmento 1: 0 - 120 s
seg1 = df[df["Tiempo(s)"] < 120]

# Segmento 2: 120 - 240 s
seg2 = df[df["Tiempo(s)"] >= 120]


plt.figure(figsize=(15,5))
plt.plot(seg1["Tiempo(s)"], seg1["Voltaje(V)"])
plt.title("ECG - Primeros 2 minutos (Reposo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)
plt.show()


plt.figure(figsize=(15,5))
plt.plot(seg2["Tiempo(s)"], seg2["Voltaje(V)"])
plt.title("ECG - Segundos 2 minutos (Lectura)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)
plt.show()

```

Identificación de los picos R y cálculo de los intervalos R-R.
```

def analizar_ecg(segmento, titulo):

    t = segmento["Tiempo(s)"].values
    ecg = segmento["Voltaje(V)"].values

    fs = 1/np.mean(np.diff(t))

    # Detectar picos R
    peaks, _ = find_peaks(
        ecg,
        distance=int(0.4*fs),
        prominence=0.3
    )

    t_R = t[peaks]
    amp_R = ecg[peaks]

    RR = np.diff(t_R)
    RR_ms = RR*1000

  
    # ECG con picos R

    plt.figure(figsize=(15,5))
    plt.plot(t, ecg)
    plt.plot(t_R, amp_R, 'ro')
    plt.title(f'Picos R - {titulo}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Voltaje (V)')
    plt.grid(True)
    plt.show()


    # Señal RR

    plt.figure(figsize=(12,4))
    plt.plot(t_R[1:], RR_ms, '-o')
    plt.title(f'Serie RR - {titulo}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('RR (ms)')
    plt.grid(True)
    plt.show()

    print(f"\n{titulo}")
    print("Latidos detectados:", len(t_R))
    print("Intervalos RR:", len(RR_ms))
    print("RR promedio =", np.mean(RR_ms), "ms")

    return t_R, RR_ms


# SEGMENTO 1
tR1, RR1 = analizar_ecg(seg1, "SEGMENTO 1")


# SEGMENTO 2
tR2, RR2 = analizar_ecg(seg2, "SEGMENTO 2")



RR_seg1 = pd.DataFrame({
    "Tiempo_R(s)": tR1[1:],
    "RR(ms)": RR1
})

RR_seg2 = pd.DataFrame({
    "Tiempo_R(s)": tR2[1:],
    "RR(ms)": RR2
})

RR_seg1.to_csv("RR_seg1.txt", sep="\t", index=False)
RR_seg2.to_csv("RR_seg2.txt", sep="\t", index=False)

```

Comparación de valores de los parámetros básicos:
```

# SEGMENTO 1
media_RR1 = np.mean(RR1)
sdnn1 = np.std(RR1, ddof=1)

# SEGMENTO 2
media_RR2 = np.mean(RR2)
sdnn2 = np.std(RR2, ddof=1)

print("SEGMENTO 1")
print("Media RR =", media_RR1, "ms")
print("SDNN =", sdnn1, "ms")

print("\nSEGMENTO 2")
print("Media RR =", media_RR2, "ms")
print("SDNN =", sdnn2, "ms")


import pandas as pd

tabla = pd.DataFrame({
    "Parámetro":["Media RR (ms)", "SDNN (ms)"],
    "Segmento 1":[media_RR1, sdnn1],
    "Segmento 2":[media_RR2, sdnn2]
})

print(tabla)

```


## Parte C 

Se construyó el diagrama de Poincaré para cada segmento de señal, representando cada intervalo R-R frente al siguiente con el fin de visualizar la dispersión de la nube de puntos en cada condición. A partir de esta representación se calcularon los índices de actividad vagal (CVI) y simpática (CSI), lo que permitió comparar el balance autonómico entre el estado de reposo y la lectura en voz alta, evidenciando cómo la verbalización genera cambios en la dinámica de la variabilidad de la frecuencia cardíaca.


Diagrama de Poincaré:
```
import matplotlib.pyplot as plt
import numpy as np


# SEGMENTO 1

RRn_1 = RR1[:-1]
RRn1_1 = RR1[1:]

plt.figure(figsize=(6,6))
plt.scatter(RRn_1, RRn1_1)
plt.xlabel("RR(n) [ms]")
plt.ylabel("RR(n+1) [ms]")
plt.title("Diagrama de Poincaré - Segmento 1")
plt.grid(True)
plt.axis('equal')
plt.show()


# SEGMENTO 2
RRn_2 = RR2[:-1]
RRn1_2 = RR2[1:]

plt.figure(figsize=(6,6))
plt.scatter(RRn_2, RRn1_2)
plt.xlabel("RR(n) [ms]")
plt.ylabel("RR(n+1) [ms]")
plt.title("Diagrama de Poincaré - Segmento 2")
plt.grid(True)
plt.axis('equal')
plt.show()

```
Indices de CVI Y de CSI

```

def calcular_cvi_csi(RR):
    RR = np.array(RR)

    RRn = RR[:-1]
    RRn1 = RR[1:]

    diff = RRn1 - RRn

    SD1 = np.sqrt(0.5) * np.std(diff, ddof=1)
    SD2 = np.sqrt(2*np.std(RR, ddof=1)**2 - 0.5*np.std(diff, ddof=1)**2)

    CSI = SD2 / SD1
    CVI = np.log10(SD1 * SD2)

    return SD1, SD2, CSI, CVI

SD1_1, SD2_1, CSI_1, CVI_1 = calcular_cvi_csi(RR1)
SD1_2, SD2_2, CSI_2, CVI_2 = calcular_cvi_csi(RR2)

tabla = pd.DataFrame({
    "Parámetro": ["SD1", "SD2", "CSI", "CVI"],
    "Segmento 1": [SD1_1, SD2_1, CSI_1, CVI_1],
    "Segmento 2": [SD1_2, SD2_2, CSI_2, CVI_2]
})

print(tabla)

```
