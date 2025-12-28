import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import random
import time

tf.random.set_seed(42)
np.random.seed(42)
random.seed(0)

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
#########################################################################################################

## Trying varying batch sizes

## Base model:
batch_size = 10
#batch_size = 50
#batch_size = 100
#batch_size = 1000

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(10),
    ]
)

print(model.summary())

# optimizer: SGD with lr = 0.01
optimizer = keras.optimizers.SGD(learning_rate=0.01)

# metrics
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
epochs = 20

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
s = time.time()
for epoch in range(epochs):
    # training loop
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 100 == 0:
            current = (step + 1) * x_batch_train.shape[0]
            size = x_train.shape[0]

    # testing loop
    test_loss = 0.0
    num_batches = 0
    test_acc_metric.reset_state()
    for x_batch_test, y_batch_test in test_ds:
        logits = model(x_batch_test, training=False)
        loss_value = loss_fn(y_batch_test, logits)
        test_loss += loss_value.numpy()
        num_batches += 1
        test_acc_metric.update_state(y_batch_test, logits)

    test_loss /= num_batches
    test_acc = test_acc_metric.result().numpy() * 100
    print(f"Test Error:\n Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f}\n")
e = time.time()
print(e-s)
print("Done!")

#################################################################################################################################

## Trying varying learning rates
batch_size = 10

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(10),
    ]
)

print(model.summary())


optimizer = keras.optimizers.SGD(learning_rate=0.01)
#optimizer = keras.optimizers.SGD(learning_rate=0.1)
#optimizer = keras.optimizers.SGD(learning_rate=0.001)

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
epochs = 20

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
s = time.time()
for epoch in range(epochs):
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 100 == 0:
            current = (step + 1) * x_batch_train.shape[0]
            size = x_train.shape[0]
    test_loss = 0.0
    num_batches = 0
    test_acc_metric.reset_state()
    for x_batch_test, y_batch_test in test_ds:
        logits = model(x_batch_test, training=False)
        loss_value = loss_fn(y_batch_test, logits)
        test_loss += loss_value.numpy()
        num_batches += 1
        test_acc_metric.update_state(y_batch_test, logits)

    test_loss /= num_batches
    test_acc = test_acc_metric.result().numpy() * 100
    print(f"Test Error:\n Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f}\n")
e = time.time()
print(e-s)
print("Done!")


