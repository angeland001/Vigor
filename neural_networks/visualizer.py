import pygame
from helpers.read_json import return_joint_cartesian
import numpy as np

#A function from a custom module (helpers.read_json) 
#that reads joint movement data from a JSON file.
movement = return_joint_cartesian(open("neural_networks/JSONs/Walking/walking_0.json"))

#convert list of movements to Numpy Array
movement = np.array(movement)

#Initialize Pygame with window
pygame.init()


size = (700, 500)

#The Pygame display surface.

screen = pygame.display.set_mode(size)
pygame.display.set_caption("Python Visualizer")

_run = True

#Pygame clock used for controlling the frame rate.
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# min_val and max_val: 
# Minimum and maximum values from the joint movement data.
min_val = np.amin(movement)
max_val = np.amax(movement)
movement -= min_val
movement *= size[1] / max_val

movement_index = 0
while _run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            carryOn = False

    screen.fill(WHITE)

    """
    This for-loop (and the movement_index) is the only code you would need to copy over/change if you want to add 
    another movement, e.g. if you wanted to visualize both the real data and the NN predictions.
    """
    for joint in movement[movement_index]:
        x, y, _ = joint
        pygame.draw.circle(screen, BLUE, [x, y], 2)

    screen.blit(pygame.transform.rotate(screen, 180), (0, 0))
    pygame.display.flip()
    clock.tick(24)

    if movement_index >= len(movement) - 1:
        movement_index = 0
    else:
        movement_index += 1

pygame.quit()
