import pygame
import numpy as np
import pandas as pd

prediction = pd.read_csv("neural_networks/networks/ae/XYZ-v2/CSVs/prediction-vidsmall-e20.csv")
prediction = np.array(prediction)[:, :-1]
prediction = prediction.reshape((len(prediction), -1, 3))

training_1 = np.array(pd.read_csv("neural_networks/networks/ae/XYZ-v2/NW-CSVs/training-2-1.csv"))[:len(prediction), :-1]
training_2 = np.array(pd.read_csv("neural_networks/networks/ae/XYZ-v2/NW-CSVs/training-2-2.csv"))[:len(prediction), :-1]
training = np.concatenate((training_1, training_2), axis=1)
training = training.reshape((len(prediction), -1, 3))

movement = np.concatenate((training, prediction), axis=1)

pygame.init()

size = (700, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Python Visualizer")

_run = True
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

min_val = np.amin(movement)
max_val = np.amax(movement)
movement -= min_val
movement *= size[1] / max_val

movement_index = 0
while _run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            carryOn = False

    screen.fill(BLACK)

    """
    This for-loop (and the movement_index) is the only code you would need to copy over/change if you want to add 
    another movement, e.g. if you wanted to visualize both the real data and the NN predictions.
    """
    for joint in movement[movement_index]:
        x, y, _ = joint
        pygame.draw.circle(screen, WHITE, [x, y], 2)

    screen.blit(pygame.transform.rotate(screen, 180), (0, 0))
    pygame.display.flip()
    clock.tick(24)

    if movement_index >= len(movement) - 1:
        movement_index = 0
    else:
        movement_index += 1

pygame.quit()
