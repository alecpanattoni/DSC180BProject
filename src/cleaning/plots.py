import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_matrix : np.array):
    """
    Plots a matrix plot of the results of generating datasets and calculating
    fairness notions. Expected input is a 2D Numpy array.

    Returns an image of the plot.
    """
    fig, ax = plt.subplots()
    im = ax.matshow(results_matrix)

    for i in range((results_matrix.shape[0])):
        for j in range((results_matrix.shape[1])):
            text = ax.text(j, i, results_matrix[i, j],
                        ha="center", va="center", color="black", fontsize=15)

    ax.set_xticklabels(['']+['insert', 'fairness', 'notions', 'here'])
    ax.set_yticklabels(['']+['No Missing', 'MCAR', 'MAR', 'NMAR'])

    plt.show()
