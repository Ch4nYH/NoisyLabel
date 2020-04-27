import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras

from datasets import make_symmetric_random_labels

from models import ResNet34

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

flags = tf.compat.v1.flags
flags.DEFINE_bool('tensorboard', True, 'Whether to save training progress')
flags.DEFINE_string('dataset', 'mnist', "What dataset to use")
FLAGS = flags.FLAGS

def load_data():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_labels = make_symmetric_random_labels(train_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels))
    
    model = ResNet34()
    
    optimizer = keras.optimizers.SGD(learning_rate = 1e-3)
    loss_fn

if __name__ == "__main__":
    main()