import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [0]
y = [0]
z = [0]

point = ax.scatter(x, y, z)

def update(frame):
    x[0] += 0.001
    y[0] += 0.001
    z[0] += 0.001

    point._offsets3d = (x, y, z)
    return point,

ani = FuncAnimation(fig, update, frames=None, blit=True, interval=1000)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()