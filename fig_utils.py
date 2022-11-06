# Author: Alexandre Allauzen
# License: MIT

import matplotlib.pyplot as plt
import os


def ax_3D(ax, preds, realdata=None,
          MIN=0, MAX=-1,
          markers=False, title=""):
    # Here we compare the prediction of
    # our model with the real points
    ax.plot3D(preds[MIN:MAX, 0],
              preds[MIN:MAX, 1],
              preds[MIN:MAX, 2],
              color='red', label='Predicted points', alpha=0.5)
    if markers:
        ax.scatter3D(preds[MIN+1:MAX, 0],
                     preds[MIN+1:MAX, 1],
                     preds[MIN+1:MAX, 2], color="red",
                     marker='x')
    # First point
    ax.scatter3D(preds[MIN, 0], preds[MIN, 1], preds[MIN, 2],
                 color="green", marker='X')
    if realdata is not None:
        ax.plot3D(realdata[MIN:MAX, 0],
                  realdata[MIN:MAX, 1],
                  realdata[MIN:MAX, 2],
                  color='blue', label='Real points',
                  alpha=0.5)
        ax.scatter3D(realdata[MIN, 0],
                     realdata[MIN, 1],
                     realdata[MIN, 2],
                     color="orange", marker='X')
    if markers:
        ax.scatter3D(realdata[MIN+1:MAX, 0],
                     realdata[MIN+1:MAX, 1],
                     realdata[MIN+1:MAX, 2],
                     color="blue",
                     marker='x')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_title(title, fontsize=15)
    return ax


def plot3D_compare(preds, realdata=None, MIN=0,
                   MAX=-1, markers=False, title=""):
    # Here we compare the prediction of our model with the real points
    _ = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax_3D(ax, preds, realdata=realdata, MIN=MIN, MAX=MAX,
          markers=markers, title=title)
    plt.legend()


def savefig(fname, format="pdf"):
    """
    Save the figure in fname (format is default = "pdf" )
    Only if the file does not already exist.
    """
    if os.path.exists(fname) and os.path.getsize(fname) > 0:
        print(fname, "already exists. Move it or change the file name")
    else:
        print("Save fig in", fname)
        plt.savefig(fname, format=format)
