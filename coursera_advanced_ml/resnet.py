import tensorflow as tf
import numpy as np
import pandas as pd


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


custom_model = Resnet(10)
x = tf.random_normal_initializer()
y = x(shape=(32, 28, 28, 1), dtype=tf.float32)
z = x(shape=(32,), dtype=tf.float32)

custom_model.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.sparse_categorical_crossentropy,
                     metrics=['mae'])


# fitting the model
cutsom_model_history = custom_model.fit(y, z, epochs=20)
