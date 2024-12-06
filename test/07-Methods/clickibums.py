import matplotlib.pyplot as plt

# Define the click event handler
def on_click(event):
    print("Hello")

# Create a figure and axes
fig, ax = plt.subplots()
ax.plot([0, 1, 2, 3], [0, 1, 4, 9])  # Example plot

# Connect the click event to the handler
fig.canvas.mpl_connect('button_press_event', on_click)

# Display the plot
plt.show()
