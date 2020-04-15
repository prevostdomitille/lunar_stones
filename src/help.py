import matplotlib.pyplot as plt


def plotter(image, title="test"):
  fig, ax = plt.subplots(1)
  ax.axis('off')
  ax.set_title(title)
  ax.imshow(image)
  plt.show()


def plotter_mult(*args, title=[]):
    nb_images = len(args)
    columns = 3
    rows = nb_images / 3
    plt.figure(figsize=(columns * 10, rows * 10))

    for i in range(0, nb_images):
        ax = plt.subplot(rows, columns, i + 1)
        if i < len(title):
            ax.title.set_text(title[i])
        plt.imshow(args[i])
    plt.show()


def to_string(img_number):
    leng = len(str(img_number))
    zeros = "0" * (4 - leng)
    return zeros + str(img_number)