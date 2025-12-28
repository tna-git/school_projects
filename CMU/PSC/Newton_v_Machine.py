#################################################################################
# PAPER HYPERPARAMETER NOTES
'''Batch size: 5,000 samples
- Training samples: Approximately 2 million data points (the full dataset)
- Architecture: 10 hidden layers with 128 ReLU units each
- Optimizer: Adam with default learning rate 0.001
- Loss: Mean squared error (MSE)
- Epochs: 1000 epochs
- Validation set: 1% of full dataset
- Mixed precision: Used for faster training (mixed_float16))'''
#################################################################################


''' ATTEMPT 1: '''
#################################################################################
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# import numpy as np

# # Load data (correct as-is)
# x = pd.read_csv("train_X.csv").values  # Shape: (N, 3) - initial conditions + time
# y = pd.read_csv("train_Y.csv").values  # Shape: (N, 4) - final positions X1x,X1y,X2x,X2y

# n = x.shape[0]
# val_size = int(0.01 * n)  # 1% validation split as per paper[attached_file:2]
# x_train, x_val = x[:-val_size], x[-val_size:]
# y_train, y_val = y[:-val_size], y[-val_size:]

# # FIXED: Dense network for vector input/output (no Conv2D/MaxPooling/Flatten)
# model = models.Sequential([
#     layers.Dense(512, activation='relu', input_shape=(x.shape[1],)),  # Input: 3 features
#     layers.Dense(512, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(y.shape[1], activation='linear')  # Output: 4 positions, linear for regression
# ])

# # Optimizer as per paper/assignment style
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam typically better for deep nets

# model.compile(
#     optimizer=optimizer,
#     loss='mse',  # Mean squared error for position regression
#     metrics=['mae']
# )

# print(model.summary())

# # Train for 1000 epochs to match Figure 3 curve[attached_file:2]
# history = model.fit(
#     x_train, y_train,
#     batch_size=64,  # Reasonable for this size
#     epochs=1000,
#     validation_data=(x_val, y_val),
#     verbose=1
# )

# # Evaluate and plot (save plot for submission)
# test_loss, test_mae = model.evaluate(x_val, y_val, verbose=0)
# print(f"Final Validation MSE: {test_loss:.6f}")
# print(f"Final Validation MAE: {test_mae:.6f}")

# plt.figure(figsize=(8, 6))
# plt.plot(history.history["loss"], label="train_loss", linewidth=2)
# plt.plot(history.history["val_loss"], label="val_loss", linewidth=2)
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.legend()
# plt.title("Training Error vs Epochs (Newton vs. The Machine)")
# plt.yscale('log')  # Log scale to match paper's Figure 3 style[attached_file:2]
# plt.grid(True, alpha=0.3)
# plt.savefig("error_curve.png", dpi=300, bbox_inches='tight')  # Save for submission
# plt.show()

# # Print final results for email body
# print("\n=== RESULTS FOR SUBMISSION ===")
# print(f"Final validation MSE: {test_loss:.8f}")
# print(f"Final validation MAE: {test_mae:.8f}")
# print(f"Graph saved as 'error_curve.png'")
####################################################################################################################

''' ATTEMPT 2: '''
# ###############################################################################################################
# import os
# import json
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, mixed_precision
# import matplotlib.pyplot as plt

# # Suppress TF warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Enable mixed precision for faster training
# mixed_precision.set_global_policy('mixed_float16')

# # Parse TF_CONFIG env variable for distributed cluster config
# tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
# task_info = tf_config.get('task', {})
# task_index = task_info.get('index', 0)

# # Define distribution strategy
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# print(f"Worker {task_index}: Running with {strategy.num_replicas_in_sync} GPUs")

# # Load dataset (should be accessible to all workers, e.g., shared filesystem)
# x = pd.read_csv("train_X.csv").values.astype(np.float32)
# y = pd.read_csv("train_Y.csv").values.astype(np.float32)

# n = x.shape[0]
# val_size = int(0.01 * n)
# x_train, x_val = x[:-val_size], x[-val_size:]
# y_train, y_val = y[:-val_size], y[-val_size:]

# # Scale batch size by number of GPU replicas
# batch_size = 2048 * strategy.num_replicas_in_sync  

# # Prepare tf.data.Dataset
# def make_dataset(x, y, batch_size, shuffle=True):
#     ds = tf.data.Dataset.from_tensor_slices((x, y))
#     if shuffle:
#         ds = ds.shuffle(10000)
#     return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# train_ds = make_dataset(x_train, y_train, batch_size)
# val_ds = make_dataset(x_val, y_val, batch_size, shuffle=False)

# # Build and compile model inside strategy scope
# with strategy.scope():
#     model = models.Sequential([
#         layers.Dense(512, activation='relu', input_shape=(x.shape[1],)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(y.shape[1], activation='linear')
#     ])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001 * strategy.num_replicas_in_sync)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# if task_index == 0:
#     print(model.summary())

# # Train model
# history = model.fit(train_ds, epochs=1000, validation_data=val_ds, verbose=2 if task_index == 0 else 0)

# # Plot results only on chief worker
# if task_index == 0:
#     plt.figure(figsize=(10, 6))
#     plt.semilogy(history.history['loss'], label='train_loss')
#     plt.semilogy(history.history['val_loss'], label='val_loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('MSE')
#     plt.legend()
#     plt.title('Newton vs The Machine - Multi-Node Training')
#     plt.savefig('error_curve_multi_node.png', dpi=300)
#     plt.show()
##############################################################################################################################

''' ATTEMPT 3: '''
###############################################################################################################################
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import matplotlib.pyplot as plt

# Mixed precision = 2x faster
mixed_precision.set_global_policy('mixed_float16')

print("GPU:", tf.config.list_physical_devices('GPU'))

# Load data
x = pd.read_csv("train_X.csv").values.astype(np.float32)
y = pd.read_csv("train_Y.csv").values.astype(np.float32)
n = x.shape[0]; val_size = int(0.01 * n)
x_train, x_val = x[:-val_size], x[-val_size:]
y_train, y_val = y[:-val_size], y[-val_size:]

# Large batch for single GPU
batch_size = 5000

# Optimized tf.data
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(val_ds)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(x.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(y.shape[1], activation='linear')  # Output layer
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
print(model.summary())

# Train 1000 epochs = ~100 minutes
history = model.fit(train_ds, epochs=1000, validation_data=val_ds, verbose=2)

# Figure 3 plot
plt.figure(figsize=(10, 6))
plt.semilogy(history.history["mae"], label="Train MAE")
plt.semilogy(history.history["val_mae"], label="Val MAE")
plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.legend(); plt.grid(True, alpha=0.3)
plt.title("Newton vs. The Machine - Single GPU")
plt.savefig("error_curve_single_gpu.png", dpi=300, bbox_inches='tight')
plt.show()

test_loss, test_mae = model.evaluate(val_ds, verbose=0)
print(f"FINAL MAE: {test_loss:.8f}")

# Playing with lr and batch size:

# FINAL MAE: 0.12127578 (batch size = int(0.0025 *n), lr = 0.01)
# FINAL MAE: 0.00000000 (batch size = 10000, lr = 0.001)
# FINAL MAE: 0.07417650 (batch size = 5000, lr = 0.001)                       <--------------- BEST MODEL