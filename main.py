import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Limiting GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Loading data into pipeline
datadir = r'C:\Imgclassification\venv\boygirl\dataTr'
data = tf.keras.utils.image_dataset_from_directory(datadir, image_size=(256, 256))

# Preprocess: Normalize the images
data = data.map(lambda x, y: (x / 255.0, y))

# Splitting the data into training, validation, and test sets
dataset_size = len(data)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = int(0.1 * dataset_size) + 1

train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size)

'''
print(f"Number of training batches: {len(train_data)}")
print(f"Number of validation batches: {len(val_data)}")
print(f"Number of test batches: {len(test_data)}")'''

# DL Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)), 
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Training
logdir = r'C:\Imgclassification\venv\boygirl\log'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])

# Plotting performance
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=18)
plt.legend(loc="upper left")
plt.show()

# Evaluating
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
rec = Recall()
B_Acc = BinaryAccuracy()

for batch in test_data.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    rec.update_state(y, yhat)
    B_Acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {rec.result().numpy()}, Accuracy: {B_Acc.result().numpy()}')

from tensorflow.keras.models import load_model
model.save(os.path.join('model','genderclassification.h5'))
#newmodel=load_model(os.path.join('model','genderclassification.h5'))