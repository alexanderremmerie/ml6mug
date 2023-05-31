#!/usr/bin/env python3

"""Model to classify mugs

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf

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

    # Define the model architecture
    model = tf.keras.Sequential([
        input_layer,  # Input layer

        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the feature maps
        tf.keras.layers.Flatten(),

        # Fully connected layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.8),

        # Output layer
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print some information about the model (number of layers, output shape, parameters, etc.)
    model.summary()

    # TODO: Return the compiled model
    return model