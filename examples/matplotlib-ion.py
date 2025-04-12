import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots()

# Initial data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create a line object
line, = ax.plot(x, y)

# Set the limits
ax.set_ylim(-1.5, 1.5)
ax.set_title("Updating Plot Example")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# Show the plot
plt.ion()  # Turn on interactive mode
plt.show()

# Update the plot in a loop
for _ in range(100):
    # Update Y data with random variation
    y = np.sin(x) + 0.1 * np.random.randn(len(x))  # Add some noise
    line.set_ydata(y)  # Update the line data

    plt.draw()  # Update the plot
    plt.pause(0.1)  # Pause to allow the plot to update visually

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final state of the plot
