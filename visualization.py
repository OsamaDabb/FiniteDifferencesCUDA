import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use('TkAgg')

# Step 1: Read the binary data from the file
filename = "output.bin"

with open(filename, "rb") as f:
    size, timesteps = np.fromfile(f, dtype=np.int32, count=2)  # Read the size (first value)
    print(size, timesteps)
    data = np.fromfile(f, dtype=np.float64)  # Read all the temperature data as floats

# Step 2: Reshape the data
# We know there are 11 timesteps, so we need to reshape the data into (timesteps, size, size, size)
#timesteps = 101
#size = 14
data_3d = data.reshape((timesteps, size, size, size))

print(f"Data shape: {data_3d.shape}")

# Assuming `data` is your 4D numpy array of shape (11, 5, 5, 5), where
# 11 represents timesteps and 5x5x5 represents the 3D grid.

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize the image object (this will be updated in each frame)
# Start with an empty plot (use the first timestep as the initial data)
img = ax.imshow(data_3d[0, :, :, data_3d.shape[3] // 2], cmap='viridis', interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(img, ax=ax, label="Temperature")
ax.set_title("Temperature at Timestep 0")

# Define the update function for each frame (timestep)
def update(frame):
    # Update the image with data from the current timestep
    slice_data = data_3d[frame, :, :, data_3d.shape[3] // 2]  # Middle slice in z-axis
    img.set_array(slice_data)
    ax.set_title(f"Timestep: {frame}")
    return [img]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=data_3d.shape[0], blit=True, interval=500)
ani.save('temperature_animation.gif', writer='Pillow', fps=30)

# Display the animation in the notebook or an interactive window
plt.show()

