# Import necessary libraries
import pygame
import numpy as np
import pandas as pd

# Read prediction data from a CSV file
prediction = pd.read_csv("neural_networks/networks/ae/XYZ-v1/CSVs/prediction-vide.csv")
prediction = np.array(prediction)[:, :-1]
prediction = prediction.reshape((len(prediction), -1, 3))

# Read training data from two CSV files and concatenate them
training_1 = np.array(pd.read_csv("neural_networks/networks/ae/XYZ-v2/NW-CSVs/training-1-1.csv"))[:len(prediction), :-1]
training_2 = np.array(pd.read_csv("neural_networks/networks/ae/XYZ-v2/NW-CSVs/training-1-2.csv"))[:len(prediction), :-1]
training = np.concatenate((training_1, training_2), axis=1)
training = training.reshape((len(prediction), -1, 3))

# Combine training and prediction data
movement = np.concatenate((training, prediction), axis=1)

# Initialize pygame
pygame.init()

# Set up the display window
size = (700, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Python Visualizer")

# Initialize variables for the main loop
_run = True
clock = pygame.time.Clock()

# Define color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Normalize the movement data to fit within the display window
min_val = np.amin(movement)
max_val = np.amax(movement)
movement -= min_val
movement *= size[1] / max_val

# Initialize index for iterating through the movement data
movement_index = 0

# Main game loop
while _run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _run = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw joints in the current frame of movement data
    for joint in movement[movement_index]:
        x, y, _ = joint
        pygame.draw.circle(screen, BLUE, [x, y], 2)

    # Rotate the screen by 180 degrees
    screen.blit(pygame.transform.rotate(screen, 180), (0, 0))
    
    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(24)

    # Update the movement index for the next iteration
    if movement_index >= len(movement) - 1:
        movement_index = 0
    else:
        movement_index += 1

# Quit pygame when the loop is exited
pygame.quit()

