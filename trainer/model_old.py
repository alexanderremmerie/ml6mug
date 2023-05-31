#!/usr/bin/env python3

"""Model to classify mugs

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf
import numpy as np
import random
import os

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 8

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 3

def set_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different mugs.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """

    # INFO: (width, height, RGB) = (64, 64, 3)
    # INFO: The model's output should be a probability for each class.
    #       The number of classes is 4.
    # INFO: The model should be ready for training after compilation.
    #       The optimizer and the loss function are to be determined.
    # INFO: We will use CNNs to solve this problem.

    set_seeds(42*9)

    # Define the model architecture
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_layer.shape[1:]),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten the feature maps
        tf.keras.layers.Flatten(),

        # Fully connected layers
        tf.keras.layers.Dropout(0.99),
        tf.keras.layers.Dense(64, activation='relu'),

        # Output layer
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes for mugs
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print some information about the model (number of layers, output shape, parameters, etc.)
    model.summary()

    # TODO: Return the compiled model
    return model