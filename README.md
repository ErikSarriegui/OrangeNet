![OrangeNet](https://github.com/ErikSarriegui/OrangeNet/assets/92121483/bfa99b7e-a1a1-4a7a-bfde-5a8ef4ac8ca4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eriksarriegui/orangenet/blob/main/web_ui.ipynb)

# **Introducción**
Este proyecto tiene como objetivo la clasificación de imágenes de naranajas según la enfermedad que esta puede sufrir (o nada en el caso de que no se detecte ningún tipo de enfermedad).

# **Instalación**
Para poder utilizar este repositorio, primero deberá clonarlo.
``` bash
$ git clone https://github.com/ErikSarriegui/OrangeNet.git
```

# **QuickStart**
## 1.1 Utilizando `web_ui.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eriksarriegui/orangenet/blob/main/web_ui.ipynb)
Puede utilizar el cuaderno Google Colab que hemos preparado para poder probar el modelo con sus propias imágenes. Únicamente debe ejecutar el cuaderno de Colab y podrá acceder a un enlace a una interfaz creada con [Gradio](https://www.gradio.app/) en la que podrá utilizar el modelo.

## 1.2 Utilizando el Pipeline
Para poder utilizar el Pipeline, primero deberá clonar el repositorio. Después podrá utilizar la clase `pipeline.OrangePipeline()` para poder realizar sus predicciones. Esta clase dispone de un método `self.inference()` (además de un método privado `self.__inference()`) que le permitirá clasificar naranjas. Este método permite que la imagen de entrada sea un `np.ndarray` así como un `PIL.Image` e incluso un `str` que sea el path a la imagen.

```python
from pipeline import OrangePipeline

model = OrangePipeline()
preds = model.inference('/path/a/imagen' | PIL.Image | np.ndarray)
print(preds)
```

# **Modelo**
Actualmente se utiliza un modelo ResNet50 en solitario que obtiene un >97% de precisión sobre los datos de testeo. Sin embargo, por la tipología del problema, en el caso de querer implementar el modelo en un caso de uso real, sería recomendable complementarlo con un modelo de segmentación previo.

# **Tutorial en profundidad**
PROXIMAMENTE

# **Dataset**
El dataset se puede encontrar en Kaggle [Kaggle](https://www.kaggle.com/datasets/jonathansilva2020/orange-diseases-dataset) y se hace referencia al siguiente [artículo](https://www.researchgate.net/publication/351229211_IDiSSC_Edge-computing-based_Intelligent_Diagnosis_Support_System_for_Citrus_Inspection).

# **Licencia**
Este repositorio queda sujeto a la licencia del dataset utilizado para entrenar el modelo.
