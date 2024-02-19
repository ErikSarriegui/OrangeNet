![OrangeNet](https://github.com/ErikSarriegui/OrangeNet/assets/92121483/bfa99b7e-a1a1-4a7a-bfde-5a8ef4ac8ca4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eriksarriegui/orangenet/blob/main/web_ui.ipynb)

# **Introducción**
Este proyecto tiene como objetivo la clasificación de imágenes de naranjas para identificar la enfermedad que las afecta, si es que la hay. En caso de que no se detecte ningún tipo de enfermedad, la imagen se clasificará como "sana".

# **Instalación**
Para poder utilizar este repositorio, primero deberá clonarlo.
``` bash
$ git clone https://github.com/ErikSarriegui/OrangeNet.git
```

# **QuickStart**
## 1.1 Utilizando `web_ui.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eriksarriegui/orangenet/blob/main/web_ui.ipynb)
Prueba el modelo con tus propias imágenes

Para que puedas probar el modelo con tus propias imágenes, hemos creado un cuaderno de Google Colab. Sigue estos pasos:

1. Accede al siguiente enlace: [Insertar enlace al cuaderno de Google Colab]
2. Ejecuta el cuaderno.
3. Obtén el enlace a la interfaz de Gradio. El enlace se mostrará en la última celda del cuaderno.
4. Abre el enlace en un navegador web.
5. Sube tus imágenes y el modelo las clasificará automáticamente.

## 1.2 Utilizando el Pipeline
Para utilizar el pipeline, siga estos pasos:
1. Importar la clase `OrangePipeline`:
```python
from pipeline import OrangePipeline
```
2. Crear una instancia de la clase: 

```python
pipeline = OrangePipeline()
```

3. Realizar una predicción:
```python
# Imagen como `np.ndarray`
imagen = np.array(...)
prediccion = pipeline.inference(imagen)

# Imagen como `PIL.Image`
imagen = Image.open(...)
prediccion = pipeline.inference(imagen)

# Imagen como ruta de acceso
ruta_imagen = "/ruta/a/imagen.jpg"
prediccion = pipeline.inference(ruta_imagen)
```
## **El método inference**
Permite clasificar imágenes de naranjas.
Admite como entrada un np.ndarray, un PIL.Image o una ruta de acceso a la imagen (str).
Devuelve una predicción con la clase de la imagen y la probabilidad.

### **Ejemplo de uso:**
```python
# Cargar la imagen
imagen = Image.open("imagen.jpg")

# Crear una instancia del pipeline
pipeline = OrangePipeline()

# Realizar la predicción
prediccion = pipeline.inference(imagen)
print(prediccion)
```

# **Modelo**
El modelo actual es un ResNet50 que obtiene una precisión superior al 99% en el conjunto de datos de prueba. Sin embargo, para un caso de uso real, se recomienda complementarlo con un modelo de segmentación previo, debido a la tipología del problema.

## Errores y confusiones
Aunque el modelo puede confundir algunas enfermedades (<1%) entre sí, nunca ha confundido una naranja enferma con una sana.

## Entrenamiento
El entrenamiento del modelo se realizó en una Nvidia Tesla T4 durante aproximadamente 87 segundos.

## **Rendimiendo del modelo**
| Modelo   | Precisión  | Nº Parámetros  |
|---|---|---|
| ResNet50  | 99,36%  | 24,033,604  |

## **Confusion Matrix**
![confusion_matrix](https://github.com/ErikSarriegui/OrangeNet/assets/92121483/3a327835-a3ac-4f11-9ec7-af06eeb0e9ef)

## **Segmentación para mejorar el modelo**
Aunque el modelo ResNet50 por sí solo ya ofrece una alta precisión, la adición de un modelo de segmentación previo podría mejorar aún más el rendimiento, ya que, en un entorno de producción, el fondo no siempre será uniforme  y las condiciones de iluminación pueden variar considerablemente. Un modelo de segmentación puede ayudar a eliminar el impacto de estos factores externos, permitiendo al modelo ResNet50 enfocarse en las características relevantes de la naranja para una clasificación más precisa.

# **Tutorial en profundidad**
**[data_setup.py:](https://github.com/ErikSarriegui/OrangeNet/blob/main/data_setup.py)**
El script data_setup.py contiene una única función, `crear_dataloaders`, que se encarga de crear los `torch.utils.data.DataLoaders` necesarios para entrenar al modelo.

Funcionalidades:
 * Carga los datasets de entrenamiento y test como objetos `torchvision.datasets.ImageFolder`.
 * Aplica las transformaciones especificadas en `transforms` a las imágenes de los datasets.
 * Crea dos `torch.utils.data.DataLoaders`, uno para entrenamiento y otro para test.
Resultado:
 * Devuelve una tupla con:
    - `train_dataloader`: DataLoader para el conjunto de entrenamiento.
    - `test_dataloader`: DataLoader para el conjunto de test.
    - `class_names`: Lista con las clases a predecir (nombres de las carpetas del dataset).

**[download_data.ipynb:](https://github.com/ErikSarriegui/OrangeNet/blob/main/download_data.ipynb)**
Funcionalidades:
 * Descarga el conjunto de datos "Orange Diseases Dataset" de Kaggle.
 * Descomprime el archivo `.zip` descargado en la carpeta data.
 * Elimina el archivo `.zip` original.
Detalles:
  * Autentica la API de Kaggle usando `api.authenticate()` por lo que es importante tener el archivo `.env` con tus credenciales.
Resultado:
  * Se crea la carpeta `data` con el conjunto de datos descomprimido.
  * Se elimina el archivo `orange-diseases-dataset.zip`.

**[engine.py:](https://github.com/ErikSarriegui/OrangeNet/blob/main/engine.py)**
Funcionalidades:
 * Entrenamiento y testeo de un modelo de clasificación de naranjas con Visión Artificial.
Contiene 3 funciones:
 * `train_step`: Realiza un paso de entrenamiento sobre un batch de datos.
 * `test_step`: Realiza un paso de testeo sobre un batch de datos.
 * `train`: Agrupa las funciones anteriores para realizar un entrenamiento completo con testeo en cada epoch.
Detalles:
 * Implementa un bucle de entrenamiento y testeo con `tqdm`.
 * Utiliza mixed precision training con `torch.cuda.amp`.
 * Calcula y guarda la pérdida y la precisión en cada epoch para entrenamiento y testeo.
Resultado:
 * Devuelve un diccionario con las listas de pérdida y precisión para poder hacer plots.

**[model.py:](https://github.com/ErikSarriegui/OrangeNet/blob/main/model.py)**
Funcionalidades:
 * Carga el modelo ResNet50 pre-entrenado con pesos por defecto.
 * Congela las capas convolucionales del modelo para evitar que se actualicen durante el entrenamiento.
 * Modifica la última capa del modelo (la "cabeza") para que tenga el número de clases correcto para la tarea de clasificación de naranjas.

Detalles:
 * La función cargar_ResNet50 toma como argumento el número de clases a predecir (out_features).
 * La cabeza del modelo se modifica reemplazando la última capa por un módulo `nn.Sequential` que consiste en:

Resultado:
 * La función devuelve una instancia del modelo `torch.nn.Module` que representa el modelo ResNet50 con la cabeza modificada.

**[pipeline.py:](https://github.com/ErikSarriegui/OrangeNet/blob/main/pipeline.py)**
Funcionalidades:
 * Carga el último modelo ResNet50 guardado en la carpeta `models`.
 * Carga las etiquetas del modelo desde el archivo `labels.json`.
 * Define un preprocesamiento de imágenes usando transformaciones de torchvision``.

Métodos:
 * `inference`(img): Recibe una imagen (`path`, `np.ndarray` o `PIL.Image`) y devuelve un diccionario con las probabilidades de cada clase de naranja.
 * `__inference`(img_tensor): Método interno que realiza la inferencia con el modelo cargado y procesa el resultado.

Ejemplo de uso:
```python
from model import OrangePipeline

# Ruta a la imagen
ruta_imagen = "imagen_naranja.jpg"

# Cargar el pipeline
pipeline = OrangePipeline()

# Predecir la clase de la naranja
prediccion = pipeline.inference(ruta_imagen)

# Imprimir las probabilidades de cada clase
for label, probabilidad in prediccion.items():
    print(f"{label}: {probabilidad}")
```

# **Dataset**
El dataset se puede encontrar en Kaggle [Kaggle](https://www.kaggle.com/datasets/jonathansilva2020/orange-diseases-dataset) y se hace referencia al siguiente [artículo](https://www.researchgate.net/publication/351229211_IDiSSC_Edge-computing-based_Intelligent_Diagnosis_Support_System_for_Citrus_Inspection).

# **Licencia**
Este repositorio queda sujeto a la licencia del dataset utilizado para entrenar el modelo.
