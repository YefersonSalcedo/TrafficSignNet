# TrafficSignNet - Clasificador de Señales de Tránsito

Un clasificador inteligente de señales de tránsito alemanas implementado con Transfer Learning utilizando ResNet50 pre-entrenada en ImageNet.

## Descripción

TrafficSignNet es un sistema de clasificación de señales de tránsito utilizando deep learning. El modelo utiliza la arquitectura ResNet50 pre-entrenada y la adapta para clasificar 43 tipos diferentes de señales de tránsito del dataset GTSRB (German Traffic Sign Recognition Benchmark).

## Características del Dataset

- **Total de imágenes**: 39,209 imágenes de entrenamiento
- **Número de clases**: 43 tipos de señales de tránsito
- **Organización**: Estructura de carpetas por clase
- **Fuente**: [GTSRB - German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Descarga de Modelos Pre-entrenados

Los modelos entrenados y archivos de configuración están disponibles en Google Drive:

**[Descargar Modelos y Configuración](https://drive.google.com/drive/folders/1VhpfUbm3aIpbQ0h_fYqJ8iJnjLEHyLqc?usp=sharing)**

Esta carpeta incluye:
- `mejor_modelo.keras` - Mejor modelo guardado durante el entrenamiento
- `modelo_final.keras` - Modelo final después de 5 épocas
- `class_names.json` - Mapeo de las 43 clases de señales

> **Nota**: Descarga estos archivos y colócalos en la raíz del proyecto para utilizar el modelo sin necesidad de re-entrenar.

## Tecnologías Utilizadas

- **TensorFlow/Keras**: Framework principal para deep learning
- **ResNet50**: Modelo pre-entrenado de 50 capas
- **Python 3.8+**
- **NumPy**: Operaciones numéricas
- **Matplotlib**: Visualización de resultados

## Estructura del Proyecto

```
TrafficSignNet/
│
├── dataset/
│   ├── Train/                      # Imágenes de entrenamiento (39,209 imágenes)
│   ├── Test/                       # Imágenes de prueba
│   └── Meta/                       # Imágenes de referencia por clase
│
├── traffic_sign_classifier.ipynb   # Notebook principal con todo el código
├── mejor_modelo.keras              # Mejor modelo guardado durante entrenamiento
├── modelo_final.keras              # Modelo final entrenado
├── class_names.json                # Nombres de las 43 clases
└── README.md                       # Este archivo
```

### Componentes:

1. **Base ResNet50**
   - Pre-entrenada en ImageNet (1.4M imágenes, 1000 clases)
   - Pesos congelados para aprovechar características aprendidas
   - Extracción de características de alto nivel

2. **GlobalAveragePooling2D**
   - Reduce dimensionalidad espacial
   - Convierte mapas de características en vector 1D
   - Previene overfitting

3. **Capa Dense Final**
   - 43 neuronas (una por cada tipo de señal)
   - Activación softmax para probabilidades
   - Clasificación multi-clase

## Rendimiento

Después de 5 épocas de entrenamiento:

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 99.55% |
| **Loss** | 0.0161 |
| **Tiempo por época** | ~70 minutos |

## Inicio Rápido

### Opción 1: Usar Modelo Pre-entrenado

```python
import tensorflow as tf
import json

# 1. Descargar archivos de Google Drive (enlace arriba)

# 2. Cargar el modelo
model = tf.keras.models.load_model("mejor_modelo.keras")

# 3. Cargar nombres de clases
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# 4. Realizar predicciones
# (ver sección de Funcionalidades Principales)
```

### Opción 2: Entrenar desde Cero

Ejecutar el notebook `traffic_sign_classifier.ipynb` completo.

## Funcionalidades Principales

### 1. Guardado Automático del Mejor Modelo

```python
checkpoint = ModelCheckpoint(
    "mejor_modelo.keras",
    save_best_only=True,
    monitor="loss",
    mode="min",
    verbose=1
)
```

### 2. Entrenamiento Continuado

Para continuar el entrenamiento desde un modelo guardado:

```python
# Cargar modelo existente
model = tf.keras.models.load_model("mejor_modelo.keras")

# Continuar entrenamiento desde época 5
history = model.fit(
    train_generator,
    epochs=10,
    initial_epoch=5,
    callbacks=[checkpoint]
)
```

### 3. Sistema de Prueba Automático

```python
prueba_completa(model, carpeta_pruebas, carpeta_meta, class_names)
```

- Selecciona aleatoriamente una imagen del conjunto de prueba
- Realiza la predicción utilizando el modelo entrenado
- Muestra comparación visual con la imagen de referencia

### 4. Visualización de Resultados

Muestra una comparación lado a lado:
- **Izquierda**: Imagen de prueba con nombre de archivo
- **Derecha**: Señal de referencia con clase predicha

## Parámetros de Configuración

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `IMAGE_SIZE` | (224, 224) | Tamaño requerido por ResNet50 |
| `BATCH_SIZE` | 32 | Imágenes procesadas simultáneamente |
| `EPOCHS` | 5 | Número de épocas de entrenamiento |
| `OPTIMIZER` | Adam | Optimizador con learning rate adaptativo |
| `LOSS` | categorical_crossentropy | Para clasificación multi-clase |


## Casos de Uso

- Sistemas de asistencia al conductor
- Vehículos autónomos
- Aplicaciones educativas de seguridad vial
- Auditorías de señalización vial
- Investigación en visión por computadora
