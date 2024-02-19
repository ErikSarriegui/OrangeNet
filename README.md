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
**data_setup.py:**
El script data_setup.py contiene una única función, `crear_dataloaders`, que se encarga de crear los `torch.utils.data.DataLoaders` necesarios para entrenar al modelo.

Funcionalidades:
Carga los datasets de entrenamiento y test como objetos `torchvision.datasets.ImageFolder`.
Aplica las transformaciones especificadas en `transforms` a las imágenes de los datasets.
Crea dos `torch.utils.data.DataLoaders`, uno para entrenamiento y otro para test.
Devuelve una tupla con:
`train_dataloader`: DataLoader para el conjunto de entrenamiento.
`test_dataloader`: DataLoader para el conjunto de test.
`class_names`: Lista con las clases a predecir (nombres de las carpetas del dataset). |

# **Dataset**
El dataset se puede encontrar en Kaggle [Kaggle](https://www.kaggle.com/datasets/jonathansilva2020/orange-diseases-dataset) y se hace referencia al siguiente [artículo](https://www.researchgate.net/publication/351229211_IDiSSC_Edge-computing-based_Intelligent_Diagnosis_Support_System_for_Citrus_Inspection).

# **Licencia**
Este repositorio queda sujeto a la licencia del dataset utilizado para entrenar el modelo.
