from __future__ import print_function, division

from sklearn.preprocessing import MinMaxScaler
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sys
import numpy as np
import json

scaler = MinMaxScaler
joint_scalers = []

class WGAN():
    def __init__(self):
        self.img_rows = 100
        self.img_cols = 24
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()

        # For the combined model we will only train the generator
        self.critic.trainable = False

        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)



        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512 * 6 * 25, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((25, 6, 512)))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))


        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        global scaler, joint_scalers
        # Load the dataset
        DATA_DIR = 'neural_networks/networks/ae/XYZ-v2/CSVs/training.csv'
        data = open(DATA_DIR, 'r').readlines()

        dataset = []
        for line in data:
            line = line.split(',')[:-4]
            for i in line:
                dataset.append(float(i))

        X = np.reshape(dataset, (-1, 72))
        X = np.transpose(X)

        new_X = []
        for joint_data in X:
            this_joint_mms = MinMaxScaler((0, 1), False)
            joint_data = np.reshape(joint_data, (-1, 1))
            this_joint_mms.fit_transform(joint_data)
            joint_scalers.append(this_joint_mms)
            new_X.append(joint_data)
        X = np.array(new_X)
        X = np.reshape(X, (72, -1))
        X = np.transpose(X)

        X_train = np.reshape(X, (-1, 100, 24, 3))

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        global scaler, joint_scalers
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = np.transpose(gen_imgs)

        transformed = []
        for idx in range(gen_imgs.shape[0]):
            reshaped = np.reshape(gen_imgs[idx], (-1, 1))
            transformed.append(joint_scalers[idx].inverse_transform(reshaped))
        transformed = np.reshape(transformed, (72, -1))

        filtered = []
        for joint in transformed:
            filtered.append(savgol_filter(joint, window_length=9, polyorder=2))
        filtered = np.array(filtered)
        filtered = np.transpose(filtered)
        np.savetxt(F'wgan-{epoch}.csv', filtered, delimiter=',', newline="\n", fmt='%10.5g')


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=10000, batch_size=8, sample_interval=120)
