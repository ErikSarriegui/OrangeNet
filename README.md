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
3. 

```python
from pipeline import OrangePipeline

model = OrangePipeline()
preds = model.inference('/path/a/imagen' | PIL.Image | np.ndarray)
print(preds)
```

# **Modelo**
Actualmente se utiliza un modelo ResNet50 en solitario que obtiene un >99% de precisión sobre el dataset de testeo. Sin embargo, por la tipología del problema, en el caso de querer implementar el modelo en un caso de uso real, sería recomendable complementarlo con un modelo de segmentación previo. Cabe resaltar que, aunque el modelo haya podido confudir alguna (<1%) enfermedad con otra, en ningún momento ha confundido una naranja enferma con una fresca. El entrenamiento se ha realizado sobre una Nvidia Tesla T4 y durante, aproximadamente, 87 segundos.

## **Rendimiendo del modelo**
| Modelo   | Precisión  | Nº Parámetros  |
|---|---|---|
| ResNet50  | 99,36%  | 24,033,604  |

## **Confusion Matrix**
![confusion_matrix](https://github.com/ErikSarriegui/OrangeNet/assets/92121483/3a327835-a3ac-4f11-9ec7-af06eeb0e9ef)

# **Tutorial en profundidad**
PROXIMAMENTE

# **Dataset**
El dataset se puede encontrar en Kaggle [Kaggle](https://www.kaggle.com/datasets/jonathansilva2020/orange-diseases-dataset) y se hace referencia al siguiente [artículo](https://www.researchgate.net/publication/351229211_IDiSSC_Edge-computing-based_Intelligent_Diagnosis_Support_System_for_Citrus_Inspection).

# **Licencia**
Este repositorio queda sujeto a la licencia del dataset utilizado para entrenar el modelo.
