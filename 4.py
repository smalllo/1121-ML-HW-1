import numpy as np
import matplotlib.pyplot as plt

# Define the vectors
w1 = np.array([1, 2, 3])
w2 = -w1

# Create a 2D plane by selecting two orthogonal vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Project the vectors onto the 2D plane
w1_2d = np.array([np.dot(w1, v1), np.dot(w1, v2)])
w2_2d = np.array([np.dot(w2, v1), np.dot(w2, v2)])

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the vectors
ax.quiver(0, 0, w1_2d[0], w1_2d[1], angles='xy', scale_units='xy', scale=1, label='w=[1,2,3]^T', color='pink')
ax.quiver(0, 0, w2_2d[0], w2_2d[1], angles='xy', scale_units='xy', scale=1, label='w=-[1,2,3]^T', color='blue')

# Set axis limits
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Show the plot
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
