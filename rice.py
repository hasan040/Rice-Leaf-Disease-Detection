import os

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
# from tensorflow.keras.layers import BatchNormalization

CHANNELS = 3
EPOCHS = 12
EPOCHS_FINAL = 20
BATCH_SIZE = 24
IMAGE_SIZE = 224


image_folder_init = "C:/Users/User/Documents/rice diseases/SYMPTOM_Resized"
image_folder = "C:/Users/User/Documents/rice diseases/Resized_Original_Data"

# dataset loading for the 17 classes
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_folder_init, shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE
)
# dataset loading for the 9 classes
dataset_final = tf.keras.preprocessing.image_dataset_from_directory(
    image_folder, shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE
)

class_names = dataset.class_names  # 17 class names
class_names_final = dataset_final.class_names  # 9 class names

print(class_names)
print("total classes", len(class_names))
print(len(dataset))

# data splitting for the 17 classes
boundary = int(len(dataset)*0.8)
train_dataset = dataset.take(boundary)
test_dataset = dataset.skip(boundary)
boundary2 = int(len(dataset)*0.1)
val_dataset = test_dataset.take(boundary2)
test_dataset = test_dataset.skip(boundary2)

# data splitting for the 9 classes
boundary = int(len(dataset_final)*0.8)
train_dataset_final = dataset_final.take(boundary)
test_dataset_final = dataset_final.skip(boundary)
boundary2 = int(len(dataset_final)*0.1)
val_dataset_final = test_dataset_final.take(boundary2)
test_dataset_final = test_dataset_final.skip(boundary2)

# caching for gpu 17 classes
train_dataset = train_dataset.cache().shuffle(300).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().shuffle(300).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().shuffle(300).prefetch(buffer_size=tf.data.AUTOTUNE)

# caching for gpu 9 classes
train_dataset_final = train_dataset_final.cache().shuffle(300).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset_final = val_dataset_final.cache().shuffle(300).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset_final = test_dataset_final.cache().shuffle(300).prefetch(buffer_size=tf.data.AUTOTUNE)

# resize & rescale for both 17 & 9
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)

])
# augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.1)
])
# applying augmentation for 17 classes training set
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# applying augmentation for 9 classes training set
train_dataset_final = train_dataset_final.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)  # 17 classes
n_classes_final = len(class_names_final)  # 9 classes

# initial model built for 17 classes training
model = models.Sequential([
    resize_and_rescale,

    layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(24, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # layers.Dropout(0.25),

    layers.Conv2D(48, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),

    layers.Dense(128),
    layers.Dropout(0.3),
    layers.Activation('relu'),

    layers.Dense(128, activation='relu'),
    # layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])


model.build(input_shape=input_shape)
print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
print("\n:::::::::::::::::::::::APPLIED ON INITIAL DATASET WITH 17 CLASSES:::::::::::::::::::::\n")
history = model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    validation_data=val_dataset,
    verbose=1,
    epochs=EPOCHS,
)
# end of first stage of training for 17 classes
# modifying the model for the second stage of training for 9 classes
model.pop()  # popping the last layer
model.add(layers.Dense(n_classes_final, activation='softmax'))  # adding the new dense layer
model.build(input_shape=input_shape)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
print("\n:::::::::::::::::::::::APPLIED ON NEW DATASET WITH 9 CLASSES:::::::::::::::::::::\n")

history1 = model.fit(
    train_dataset_final,
    batch_size=BATCH_SIZE,
    validation_data=val_dataset_final,
    verbose=1,
    epochs=EPOCHS_FINAL,
)

for images_batch, labels_batch in test_dataset_final.take(6):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    print("first image to predict")
    print("actual label:", class_names_final[first_label])
    batch_prediction = model.predict(images_batch)
    print("predicted label:", class_names_final[np.argmax(batch_prediction[0])])
    print("...........||............")


# model_version = max([int(i) for i in os.listdir("model") + [0]])+1
# model.save(f"model/{model_version}")









