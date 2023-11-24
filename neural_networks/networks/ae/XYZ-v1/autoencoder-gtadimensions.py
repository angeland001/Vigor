# Import necessary modules and libraries
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import TensorBoard

# Task: Open and read the training and target files
#HINT: Open up csv files in both training and test csv found in CSV Folder
#HINT: Initialze empty lists for training and testing

# Task: Read data from the training file
#HINT: Using a loop, go through every row (Basically .readlines())
    #using another loop (Nested Loop), for each item in that row with it being seperated by .split(",")
    #Check to see if the item in each iteration is **not** a new line or empty
        #If it isnt, append that data to training list
    #Check for ValueError and just pass it to be sure (GOOGLE)

'''
# Task: Read data from the Target file (Same as above)
HINT: Using a loop, go through every row (Basically .readlines())
    #using another loop (Nested Loop), for each item in that row with it being seperated by .split(",")
    #Check to see if the item in each iteration is **not** a new line or empty
        #If it isnt, append that data to training list
    #Check for ValueError and just pass it to be sure (GOOGLE)


'''

'''

# Task: Reshape and preprocess the data
Initilize new variables for reshaping the previous made lists

The np.reshape function is used to reshape the training and target data arrays.
(-1, 18, 1) specifies the desired shape of the arrays. Here, -1 means that the size of that dimension is inferred based on the length of the data, 
and 18 and 1 represent the dimensions. The data is reshaped to have a three-dimensional structure where the first dimension represents the number 
of samples (possibly the number of frames or instances), the second dimension represents the sequence length (here, 18), and the third dimension represents the number of features (here, 1).
'''

#Hint: Look up reshape function on tensorflow and apply it to newly created variables to apply it from training and testing variables above
# After that, convert the typing of each to floating point.


# Task: Define the architecture of the autoencoder model
# Hint: Define an input layer with the specified shape
input_img = Input(shape=(18, 1))

# Task: Implement the encoder layers
# Hint: Use Conv1D layers to learn hierarchical features, followed by MaxPooling for downsampling, Lastly encode it using maxPooling. each layer is multiplied from the previous layer
# Another hint: GOOGLE! You have to learn what these do lol.
# also look at DAX machine learning tutorials for decoding and encoding

# Task: Implement the decoder layers
# Hint: Use Conv1D layers for upsampling, followed by UpSampling1D layers. Lastly decode it using Conv1D. each layer is multiplied from the previous layer


# Task: Create the autoencoder model
# Hint: Use the Model class to define the model by specifying input and output layers
autoencoder = Model(input_img, decoded)

# Task: Compile the model with an optimizer and loss function
# Hint: Choose an optimizer and specify the loss function
autoencoder.compile(optimizer='adam', loss='mse')

# Task: Train the autoencoder model
# Hint: Use the fit method, specify the training data and parameters
autoencoder.fit(x_train, x_test,
                epochs=1000,
                batch_size=32,
                shuffle=True,
                validation_data=(x_train, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

# Task: Save the trained autoencoder model to a file
# Hint: Use the save method of the model
autoencoder.save("Models/dae-xyz-gta.h5")

