# Hybrid Rotation Invariant Network

This is a **Tensorflow**/**Keras** implementation of the rotation invariant layers introduced in [Hybrid Rotation Invariant Networks for small sample size deep learning](https://openreview.net/pdf?id=BJlVNY8llV) by *Alexander Katzmann, Marc-Steffen Seibel, Alexander Mühlberg, Michael Sühling, Dominik Nörenberg, Stefan Maurus, Thomas Huber* and *Horst-Michael Groß*.

## Usage
The implemented layers can be used as standard layers within the Keras network and follow the basic implementation of *keras.layers.Layer*. They can thus be used like standard Conv2D layers, including functions for initialization, serialization, etc., providing a similar API. A basic example might look like the following:

```python
import keras

from keras.layers import Input, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.models import Model

from layers import RotationalConv2D

num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = [keras.utils.to_categorical(el, num_classes) for el in [y_train, y_test]]

x = inp = Input((32,32,1))

x = RotationalConv2D(32, kernel_size=(3,3), strides=(1,1), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = RotationalConv2D(64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = RotationalConv2D(128, kernel_size=(3,3), strides=(1,1), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Flatten()(x)

x = Dense(num_classes, activation='softmax')(x)
out = x

model = Model([inp],[out])

model.compile(
   optimizer = keras.optimizers.Adam(),
   loss      = keras.losses.categorical_crossentropy,
   metrics   = ['acc']
)

model.fit(
   [x_train],
   [y_train],
   validation_data = [[x_test],[y_test]],
   epochs = 10
)
```

## Transforming Conv2D-networks in Hybrid Rotation Invariant Networks
Hybrid networks utilizing both, standard and classical convolutions, can be realized using a mixture of classical and rotationally invariant Conv2D layers. Transforming a Conv2D-based network into a hybrid rotation invariant network as proposed in the forementioned paper can easily be done by prepending the following code snippet:

```python
class Conv2D():
    def __init__(self, n_filters, **kwargs):
        self.n_filters = n_filters
        self.kwargs = kwargs
        self.nkwargs = dict(self.kwargs)
        
        if not "effective_kernel_size" in self.nkwargs:
            self.nkwargs["effective_kernel_size"] = self.nkwargs["kernel_size"]
        
    def __call__(self, x):
        
        origConv = keras.layers.Conv2D(self.n_filters // 2, **self.kwargs)(x)
        nConv    = RotationalConv2D.RotationalConv2DPose(self.n_filters // 2 - 2, **self.nkwargs)(x)
            
        return Concatenate()([origConv, nConv])
```
