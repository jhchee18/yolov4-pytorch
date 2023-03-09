import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pygame
from time import sleep
import math

# Create a figure and add a 3D axis to it
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Set the axis limits to be -1 and 1, with an interval of 1
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xticks(np.arange(-100, 100, 20))
ax.set_yticks(np.arange(-100, 100, 20))
ax.set_zticks(np.arange(-100, 100, 20))

# Create an array to store the x, y, and z coordinates of the points
n_points = 100
xs = np.zeros(n_points)
ys = np.zeros(n_points)
zs = np.zeros(n_points)

# Use the scatter method to create a scatter plot of the points
points = ax.scatter(xs, ys, zs)

ax.set_xlabel('X-Axis (cm)')
ax.set_ylabel('Y-Axis (cm)')
ax.set_zlabel('Z-Axis (cm)')


# Initialize Pygame
pygame.init()

# Create a display surface
screen = pygame.display.set_mode((640, 480))

### PARAMETERS ###
fSpeed = 117 / 10 # Forward Speed in cm/s
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval = fSpeed * interval
aInterval = aSpeed * interval

x, y = 500, 500
a = 0
yaw = 0

# Define a function to update the position of the points
def update_position():
    global xs, ys, zs

    global x, y, yaw, a
    d = 0
        
    # Shift the x, y, and z coordinates of the points by one index
    xs[:-1] = xs[1:]
    ys[:-1] = ys[1:]
    zs[:-1] = zs[1:]

    # Check which arrow keys are currently pressed
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        d = dInterval
        a = -180
        #xs[-1] += -0.1 # move left in x-axis

    elif keys[pygame.K_RIGHT]:
        d = -dInterval
        a = 180
        #xs[-1] += 0.1 # move right in x-axis



    if keys[pygame.K_UP]:
        d = -dInterval
        a = -90
        #zs[-1] += 0.1 # move up in z-axis

    elif keys[pygame.K_DOWN]:
        d = dInterval
        a = 270
        #zs[-1] += -0.1 # move down in z-axis



    if keys[pygame.K_w]:
        zs[-1] += dInterval # move up in z-axis
    elif keys[pygame.K_s]:
        zs[-1] += -dInterval # move down in z-axis



    if keys[pygame.K_a]:
        yaw += aInterval
        #zs[-1] += 0.1 # move up in z-axis

    elif keys[pygame.K_d]:
        yaw -= aInterval
        #zs[-1] += -0.1 # move down in z-axis


    """
    # Check for key presses and update coordinates accordingly
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                zs[-1] += 0.1  # move up in z-axis
            elif event.key == pygame.K_s:
                zs[-1] -= 0.1  # move down in z-axis
            elif event.key == pygame.K_a:
                xs[-1] -= 0.1  # move left in x-axis
            elif event.key == pygame.K_d:
                xs[-1] += 0.1  # move right in x-axis
    """

    sleep(interval)
    a += yaw
    xs[-1] += int(d * math.cos(math.radians(a)))
    ys[-1] += int(d * math.sin(math.radians(a)))

    # Update the position of the scatter plot with the new coordinates
    points._offsets3d = (xs, ys, zs)


# Continuously update the position of the points and redraw the plot
while True:
    update_position()
    plt.pause(0.1)
    plt.draw()

# Show the plot
plt.show()
