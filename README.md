![logo](https://github.com/ErikSarriegui/OrangeNet/assets/92121483/b8e18ac1-8d33-4b56-b6ba-c49315114ae3)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<usuario>/<repositorio>/blob/main/<nombre_del_archivo>.ipynb)
# **Introducción**
Este proyecto tiene como objetivo la clasificación de imágenes de naranajas según la enfermedad que esta puede sufrir (o nada en el caso de que no se detecte ningún tipo de enfermedad).

# **Instalación**
Para poder utilizar este repositorio, primero deberá clonarlo.
```
$ git clone https://github.com/ErikSarriegui/OrangeNet.git
´´´

# **QuickStart**
## 1.1 Utilizando `runner.ipynb`
Habiendo clonado el repositorio, comience clonando instalando las dependencias necesarias:
```
$ pip install -r requirements.txt
```

Puede realizar las primeras predicciones ejecutando la primer celda cuaderno `runner.ipynb`. En el caso de querer intercambiar la imagen a clasificar, simplemente inserte la ruta de su imagen:
```
import predictions

predictions.inference("""Su/ruta/aquí""")
```
## 1.2 Utilizando `web_ui.ipynb`
Además, ejecutando la primera celda de web_ui, aparecerá un enlace a una interfaz creada con [Gradio](https://www.gradio.app/) en la que podrá utilizar el modelo.

# **Modelo**
Actualmente se utiliza un modelo ResNet50 en solitario que obtiene un >97% de precisión sobre los datos de testeo. Sin embargo, por la tipología del problema, en el caso de querer implementar el modelo en un caso de uso real, sería recomendable complementarlo con un modelo de segmentación previo.

# **Tutorial en profundidad**
PROXIMAMENTE

# **Dataset**
El dataset se puede encontrar en Kaggle [Kaggle](https://www.kaggle.com/datasets/jonathansilva2020/orange-diseases-dataset) y se hace referencia al siguiente [artículo](https://www.researchgate.net/publication/351229211_IDiSSC_Edge-computing-based_Intelligent_Diagnosis_Support_System_for_Citrus_Inspection).

# **Licencia**
Este repositorio queda sujeto a la licencia del dataset utilizado para entrenar el modelo.
