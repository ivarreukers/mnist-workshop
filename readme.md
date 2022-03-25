
Opzet:
--
Deze ochtend gaan we een eigen netwerk maken om handgeschreven getallen te herkennen. Hiervoor zal de bestaande MNIST dataset gebruikt worden. 
Dit is een dataset van in totaal 70.000 handgeschreven 28x28px gelabelde plaatjes . 

Opdr 1 - Omgeving optuigen
--
Stap 1 is het opzetten van de Python environment. 
Maak een virtual environment aan 

`python -m venv [path-to-venv]`

'Activeer' de venv met:
 - Mac/Linux: `$ source <venv>/bin/activate`
 - Windows CMD: `C:\> <venv>\Scripts\activate.bat`
 - Windows Powershell: `PS C:\> <venv>\Scripts\Activate.ps1`
    
Installeer de dependencies met `pip install -r requirements.txt`

Krijg je een melding "No matching distribution found for tensorflow"? Haal deze dan uit de `requirements.txt` en installeer deze 'handmatig' met:
`pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl`

Heb je een nieuwe macbook met de M1 chip? Lees dan `m1/mac-m1-tensorflow.md`

Opdr 2 - Dataset inladen
--
Als alle dependencies gedownload zijn is het tijd om de MNIST dataset in te laden. Dit kan op twee manieren:

*Manier 1:*
De ontwikkelaars van keras zijn zo vriendelijk geweest om de MNIST dataset in te bouwen in `keras-datasets`
Om de dataset in te laden kan het simpelweg geimporteerd worden:

```python
from keras.datasets import mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
```

Manier 2:
De tweede manier is om de dataset handmatig in te laden. 

```python
import os
import gzip
import numpy as np

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                            '%s-labels-idx1-ubyte.gz'
                            % kind)
    images_path = os.path.join(path,
                            '%s-images-idx3-ubyte.gz'
                            % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 784)

    return images, labels
```

Vervolgens kan `load_mnist` aangeroepen worden om de (train_data, train_labels) in te laden.

Opdr 3 - Normaliseren
--

Zet de data om naar een voor het model herkenbare shape & als float

```python
data = data.reshape((data.shape[0], height*width)).astype('float32')
```

doe dit voor zowel de train & de test dataset (alleen de data, niet voor de labels).

Vervolgens is het nodig om de labels om te zetten naar numerieke waarden voor het model. Dit kan met de `to_categorical` methode van numpy die een one-hot encoding uitvoerd. `number_of_classes` = aantal unieke labels

```python
labels = np_utils.to_categorical(labels, number_of_classes)
```

Voor deze encoding voor zowel de train- & test labels uit. 

Momenteel zijn de waardes van de pixels in onze data tussen 0 & 255. Ons netwerk werkt sneller met waardes tussen 0 & 1. 
Zet de train- & testdata om naar waardes tussen 0 & 1 (hint; type is een nparray)

Opdr 4 - Data bekijken
--
Om een voorbeeld visueel te kunnen zien kan de matplot library gebruikt worden. 

```python
import matplotlib.pyplot as plt
```
```python
# needed to reshape into 'image' of 28x28 with 1 channel (black/white) form for matplotlib instead of single array.
# remove after this step as our model expects the single array
train_data = train_data.reshape((train_data.shape[0], 28, 28, 1)).astype('float32')
print(train_labels[0])
plt.imshow(train_data[0])
plt.colorbar()
plt.show()
```
(Note; `plt.show()` is blocking)


Opdr 5 - model aanmaken
--
De omgeving is opgezet en de data is genormaliseerd, tijd om het model te bouwen.
Dit gaan we doen met het 'Sequential' model vanuit Keras met 'Dense' layers. Zie hiervoor de documentatie van keras:

https://keras.io/api/models/sequential/
https://keras.io/api/layers/core_layers/dense/ 
(`Sequential` is ook te importeren vanuit `keras.models` & `Dense` vanuit `keras.layers`)

Begin met het toevoegen van de input laag en de eerste Dense layer.
Gebruik `input_shape=(None, height*width)` of `input_dim=height*width`

Gebruik 'relu' als activatiefunctie voor de eerste laag en geef het 16 outputs. 

Voeg nog een hidden-layer toe met 16 outputs

Voeg nu de output layer toe

Compile het model met als optimizer 'Adam', loss function Mean Squared Error (mse) & accuracy als metric.

Een overzicht van het model kan geprint worden m.b.v. `model.summary()`



Opdr 6 - trainen
--
Nu het model aangemaakt is, is het tijd om te trainen.

Gebruik hiervoor de fit methode:
https://keras.io/api/models/model_training_apis/#fit-method 

Vul hiervoor de train data, labels, een aantal epochs en een batch size in. 

Als het model getraind is, kunnen we kijken hoe het model presteert op de test set

`scores = model.evaluate(test_data, test_labels)`

Opdr 7 - Confusion Matrix
--
Een andere, inzichtelijkere manier om te zien hoe het model heeft gepresteerd is een confusion matrix. 

```python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

fig = plt.figure(figsize=(10, 10))

test_preds = model.predict(test_data) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

test_predictions = np.argmax(test_preds, 1) # Decode Predicted labels
predicted_labels = np.argmax(test_labels, 1) # Decode labels

mat = confusion_matrix(predicted_labels, test_predictions) # Confusion matrix

# Plot Confusion matrix
sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
plt.show();
```

Uitleg over hoe de confusion matrix gelezen kan worden + verdere uitleg: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/ 


Wat nu?
--
De basis is klaar! Kies zelf wat je verder met het model wilt gaan doen:

 - Probeer het model te verbeteren, wie haalt de hoogste score?
    - Online zullen vaak CNNs als voorbeeld voorbij komen; die komen in het middag programma aan bod!
 - Maak het model compatible voor letters i.p.v. getallen (EMNIST dataset)
 - Maak je eigen data (28x28) en herken je eigen handschrift
 - Maak het productie-ready:
    - Schrijf het model na het trainen weg
    - Maak een simpele REST interface
    - Laadt het model in en gebruik het via de API

Meer hands-on ervaring opdoen?
Een diepere en verdergaande workshop is beschikbaar op Kaggle:
https://www.kaggle.com/learn/intro-to-deep-learning  
 
Verdere theorie
 - Loss Functions: https://towardsdatascience.com/what-is-loss-function-1e2605aeb904
 - Optimizers: https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
 - 3Blue1Brown serie over Deep Learning met MNIST: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown 
