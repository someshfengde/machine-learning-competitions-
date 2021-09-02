import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks
from tensorflow.python.ops.math_ops import DivideDelegateWithName


class IdentityBlock(tf.keras.Model):
    def __init__(self, units, kernel_size):
        super(IdentityBlock, self).__init__()
        self.lay_1 = tf.keras.layers.Conv2D(
            units, kernel_size=kernel_size, activation='relu', padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.lay_2 = tf.keras.layers.Conv2D(
            units, kernel_size=kernel_size, activation='relu', padding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.lay_1(input_tensor)
        x = self.bn_1(x)
        x = self.lay_2(x)
        x = self.bn_2(x)
        x = self.add([x, input_tensor])
        return x


class Resnet(tf.keras.Model):
    def __init__(self, num_classes):
        super(Resnet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            64, 3, padding='same', activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.id1a = IdentityBlock(units=64, kernel_size=3)
        self.id1b = IdentityBlock(units=64, kernel_size=3)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(
            num_classes, activation='softmax')

    def call(self, inputs):
        print('building model ')
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.id1a(x)
        x = self.id1a(x)
        x = self.global_pool(x)
        return self.classifier(x)


# custom_model = Resnet(10)
# x = tf.random_normal_initializer()
# y = x(shape=(32, 28, 28, 1), dtype=tf.float32)
# z = x(shape=(32,), dtype=tf.float32)

# custom_model.compile(optimizer=tf.keras.optimizers.Adam(),
#                      loss=tf.keras.losses.sparse_categorical_crossentropy,
#                      metrics=['mae'])


# # fitting the model
# cutsom_model_history = custom_model.fit(y, z, epochs=20)


# building new model with simpler layer
class Mod(tf.keras.Model):
    def __init__(self, units):
        super(Mod, self).__init__()
        self.units = units
        self.d_lay = tf.keras.layers.Dense(units, activation='linear')

    def call(self, inputs):
        x = self.d_lay(inputs)
        x = self.d_lay(x)
        return x


model = Mod(1)

model.compile(loss=tf.keras.losses.mse,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

# building custom loss function


class Custom_callback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print(f'training batch {batch} begin at {datetime.datetime.now()}')
        # return super().on_train_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        print(f'training batch {batch} ends at {datetime.datetime.now()}')
        # return super().on_train_batch_end(batch, logs=logs)


my_custom_callback = Custom_callback()
X_train = tf.constant(np.arange(0, 1000))
y_train = tf.random.uniform(shape=(1000,))
model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=0,
          callbacks=[my_custom_callback])


# making detect overfitting callback

class DetectOverFittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(DetectOverFittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_begin(self, epoch, logs='None'):
        ratio = logs['val_loss']/logs['loss']
        print(f'epoch {epoch} val_train loss raio = {ratio}')

        if ratio > self.threshold:
            print('Stopping training')
            self.model.stop_training = True

        # return super().on_epoch_begin(epoch, logs=logs)


model.fit(X_train[:500], y_train[:500], validation_data=(X_train[500:], y_train[500:]), epochs=100,
          callbacks=[DetectOverFittingCallback(1.3), my_custom_callback])


# making Visual callback
