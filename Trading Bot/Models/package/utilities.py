import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

 
# Check the prediction
def check_the_pred(df, Reality, Pred):
    Check = []
    
    for res, pred in zip(df[Reality], df[Pred]):
        if res == pred:
            Check.append(True)
        else:
            Check.append(False)
    return Check
  
# Plot
def plot_confusion_matrix(cf, classes):
    fig, ax = plt.subplots(figsize = (4,4))
    sns.heatmap(cf, annot = True, fmt='g',
                linewidths = .9, square = True, 
                cmap = 'Blues_r', cbar = False,
                xticklabels = classes,
                yticklabels = classes)

    ax.set_ylabel('True Label', fontsize = 10)
    ax.set_xlabel('Predicted Label', fontsize = 10)

def plot_error_graph(stats,  n, x_label):
    
    fig, ax = plt.subplots(figsize = (4,4))

    ax.plot(n, stats[0,:], 'o:', label = 'Error')
    ax.plot(n, stats[1,:], 'o:', label ='Bias$^2$')
    ax.plot(n, stats[2,:], 'o:', label ='Variance')
    ax.set_xlabel(x_label)
    ax.grid()
    ax.legend()
    
def plot_accuracy(L1, L2, lab1, lab2, n, x_label):
    
    fig, ax = plt.subplots(figsize = (4,4))

    ax.plot(n, L1, 'o:', label = 'Accuracy Train')
    ax.plot(n, L2, 'o:', label = 'Accuracy Valid')
    ax.set_xlabel(x_label)
    ax.grid()
    ax.legend()

