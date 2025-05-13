import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialization
is_init = False
label = []
label_dict = {}
class_counter = 0

X, y = None, None

# Load all .npy feature files except labels.npy
for filename in os.listdir():
    if filename.endswith(".npy") and filename != "labels.npy":
        emotion_label = os.path.splitext(filename)[0]
        features = np.load(filename)

        if not is_init:
            X = features
            y = np.array([[emotion_label]] * features.shape[0])
            is_init = True
        else:
            X = np.concatenate((X, features), axis=0)
            y = np.concatenate((y, np.array([[emotion_label]] * features.shape[0])), axis=0)

        label.append(emotion_label)
        label_dict[emotion_label] = class_counter
        class_counter += 1
        logging.info(f"Loaded {filename} with label '{emotion_label}'")

# Map string labels to integers
y_int = np.vectorize(label_dict.get)(y.flatten()).astype("int32")
y_cat = to_categorical(y_int)

# Shuffle data (with reproducibility)
np.random.seed(42)
shuffle_indices = np.random.permutation(X.shape[0])
X_shuffled = X[shuffle_indices]
y_shuffled = y_cat[shuffle_indices]

# Validate input dimensions
input_shape = X.shape[1]
if input_shape == 0:
    raise ValueError("Input feature size is zero. Check the input .npy files.")

# Build model
input_layer = Input(shape=(input_shape,))
x = Dense(512, activation="relu")(input_layer)
x = Dense(256, activation="relu")(x)
output_layer = Dense(y_cat.shape[1], activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model with validation split
logging.info("Starting model training...")
model.fit(X_shuffled, y_shuffled, epochs=50, batch_size=32, validation_split=0.2)
logging.info("Model training completed.")

# Save model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
logging.info("Model and label mapping saved successfully.")
