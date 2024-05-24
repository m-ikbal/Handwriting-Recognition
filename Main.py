import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, \
    GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def resnet_block(input_tensor, filters, kernel_size=3, strides=1, conv_shortcut=True):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    if conv_shortcut:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet152(input_shape=(64, 64, 3), num_classes=26):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = resnet_block(x, filters=64, strides=1, conv_shortcut=False)
    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=64)

    x = resnet_block(x, filters=128, strides=2)
    for _ in range(7):
        x = resnet_block(x, filters=128)

    x = resnet_block(x, filters=256, strides=2)
    for _ in range(35):
        x = resnet_block(x, filters=256)

    x = resnet_block(x, filters=512, strides=2)
    for _ in range(2):
        x = resnet_block(x, filters=512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


digits = load_digits()
x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=26)

model = resnet152()

x_train = np.reshape(x_train, (-1, 8, 8, 1))
x_test = np.reshape(x_test, (-1, 8, 8, 1))

x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)

x_train = tf.image.resize(x_train, [64, 64])
x_test = tf.image.resize(x_test, [64, 64])

x_train = x_train.numpy()
x_test = x_test.numpy()

optimizer = Adam(learning_rate=0.001)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[early_stopping])
