import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Read your image using OpenCV
img = cv2.imread('img11.jpg')

# Convert BGR image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Set up the main figure and axes
fig, ax = plt.subplots()

# Display the image without axis ticks and labels
ax.imshow(img_rgb)
ax.axis('off')

# Create zoomed inset axes
axins = zoomed_inset_axes(ax, 2, loc='upper right')  # Adjust the zoom level as needed

# Display the image on the zoomed inset axes
axins.imshow(img_rgb)

# Remove ticks and labels from the inset axes
axins.axis('off')

# Mark the region of the zoomed inset on the main axes
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="r")

# Maximize the window (full screen)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

# Show the plot
plt.show()
