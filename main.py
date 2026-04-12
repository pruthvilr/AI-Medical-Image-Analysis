import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2

# Image size and batch
IMG_SIZE = 224
BATCH_SIZE = 32

# Dataset paths
train_dir = "data/train"
test_dir = "data/test"

# Data preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load pre-trained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze layers
base_model.trainable = False

# Build final model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=3
)

# Save model
model.save("models/model.h5")

# Plot accuracy graph
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.savefig("outputs/accuracy.png")
plt.show()

# Confusion matrix
y_true = test_data.classes
y_pred = model.predict(test_data)
y_pred = np.round(y_pred)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Test prediction on one image
img = cv2.imread("data/test/PNEUMONIA/BACTERIA-932292-0001.jpeg")

if img is None:
    print("❌ Image not found! Check file path.")
    sys.exit()  
else:
    img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.reshape(img, (1,224,224,3))

prediction = model.predict(img)

if prediction > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")