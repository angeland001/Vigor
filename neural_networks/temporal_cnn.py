# Import necessary libraries from Keras and helpers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, InputLayer, Flatten
from helpers.read_json import return_joint_cartesian
import numpy as np

# Open a JSON file containing joint Cartesian data for a walking sequence
json_file = open("JSONs/Walking/walking_0.json")
joint_cartesian = return_joint_cartesian(json_file)

# Set the lookahead size for input data and initialize empty lists for features (X) and labels (y)
lookahead_size = 100
X = []
y = []

# Create input sequences (X) and corresponding labels (y) for training the model
for index in range(len(joint_cartesian) - (lookahead_size + 1)):
    X.append(joint_cartesian[index:index + lookahead_size])
    y.append(joint_cartesian[index + (lookahead_size + 1)])

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Reshape y to have 78 columns (assuming 26 joints with 3 coordinates each)
y = np.reshape(y, (-1, 78))

# Print the shapes of the training data
print(f"Train data shape: {X.shape}")
print(f"Test data shape: {y.shape}")

# Create a sequential model using Keras
temporal_model = Sequential()

# Add layers to the model
temporal_model.add(InputLayer(input_shape=(lookahead_size, 26, 3)))
temporal_model.add(Conv2D(32, (7, 5), activation="relu"))
temporal_model.add(Conv2D(32, (7, 3), activation="relu"))
temporal_model.add(MaxPool2D(2))
temporal_model.add(Conv2D(64, (7, 3), activation="relu"))
temporal_model.add(Conv2D(64, (7, 3), activation="relu"))
temporal_model.add(MaxPool2D(2))
temporal_model.add(Conv2D(128, (7, 3), activation="relu"))
temporal_model.add(Flatten())
temporal_model.add(Dense(1280, activation="relu"))
temporal_model.add(Dense(1280, activation="relu"))
temporal_model.add(Dense(78, activation="linear"))

# Compile the model using the Adam optimizer and mean squared error loss
temporal_model.compile(optimizer="adam", loss="mse")

# Display a summary of the model architecture
temporal_model.summary()

# Train the model using the training data (X, y) with a batch size of 16 and 1000 epochs
temporal_model.fit(X, y, batch_size=16, epochs=1000)

