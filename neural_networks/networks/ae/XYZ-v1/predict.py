from keras.models import load_model
import numpy as np
from math import acos, degrees

autoencoder = load_model(r"C:\Users\angel\Documents\GitHub\Vigor\neural_networks\networks\ae\XYZ-v1\Models\dae-xyz-gta1.h5")
training_file = open(r'C:\Users\angel\Documents\GitHub\Vigor\neural_networks\networks\ae\XYZ-v1\CSVs\target-gta.csv', 'r')

training_data = []

for line in training_file.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                training_data.append(float(data))
            except ValueError:
                pass
training_data = np.reshape(training_data, (-1, 18, 1))
prediction_data = autoencoder.predict(training_data)

prediction_file = open(r'C:\Users\angel\Documents\GitHub\Vigor\neural_networks\networks\ae\XYZ-v1\CSVs\prediction-gta.csv', 'w')
prediction_data = np.reshape(prediction_data, (-1, 18, 1))

for frame in prediction_data:
    for data in frame:
        prediction_file.write(str(data[0]))
        prediction_file.write(",")
    prediction_file.write("\n")



