import matplotlib.pyplot as plt


def plot_image(image_x):
    plt.imshow(image_x.reshape(image_x.shape[:2]))


def plot_image_z(image_x, z):
    plot_image(image_x[z, :, :])


def plot_image_y(image_x, y):
    plot_image(image_x[:, y, :])


def plot_image_x(image_x, x):
    plot_image(image_x[:, :, x])
