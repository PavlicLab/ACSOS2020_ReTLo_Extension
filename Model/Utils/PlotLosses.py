import numpy as np
import pandas as pd
import time
import os
import pathlib

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Activation, Dense, TimeDistributed,Conv2D,MaxPooling2D, Lambda,UpSampling2D
from tensorflow.keras.layers import add, dot, concatenate, LSTM, Bidirectional,Reshape,Flatten
import tensorflow.keras as keras

class PlotLosses(keras.callbacks.Callback):

    def __init__(self, targets=['loss']):

        self.targets = []
        for t in targets:
            self.targets.append(t)
            self.targets.append('val_{}'.format(t))


    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []

        for _ in range(len(self.targets)):
            self.losses.append([])
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        from IPython.display import clear_output

        # append values
        self.logs.append(logs)
        self.x.append(self.i)

        for i in range(len(self.targets)):
            self.losses[i].append(logs.get(self.targets[i]))
        self.i += 1

        # visualize
        clear_output(wait=True)
        self.fig = plt.figure(figsize=(24, 5)) # w, h

        for i in range(len(self.targets)):
            if i % 2 == 0:
                ax = self.fig.add_subplot(1, len(self.targets)//2 + 1, i//2 + 1)
            ax.plot(self.x, self.losses[i], label=self.targets[i])
            ax.legend()

        plt.show()
        
        