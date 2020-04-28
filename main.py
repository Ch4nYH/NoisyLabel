import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as layers

from datasets import make_symmetric_random_labels
from losses import WeightedCrossEntropy
from models import ResNet34

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

flags = tf.compat.v1.flags
flags.DEFINE_bool('tensorboard', True, 'Whether to save training progress')
flags.DEFINE_string('dataset', 'mnist', "What dataset to use")
FLAGS = flags.FLAGS

def load_data():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    return (train_images, train_labels), (test_images, test_labels)

def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images = train_images.reshape(60000, 784).astype('float32') / 255
    test_images = test_images.reshape(10000, 784).astype('float32') / 255
    #train_labels = make_symmetric_random_labels(train_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels))
    
    batch_size = 64
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(64)
    
    #model = ResNet34()
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.SGD(learning_rate = 1e-3)
    #loss_fn = WeightedCrossEntropy()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #meta_loss_fn = keras.losses.CategoricalCrossentropy()
    
    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy() 
    
    epochs = 3
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric(y_batch_train, logits)
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))
        
        train_acc = train_acc_metric.result()
        print('Training acc over epoch: %s' % (float(train_acc),))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        #Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val)
            val_acc_metric(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print('Validation acc: %s' % (float(val_acc),))
    

if __name__ == "__main__":
    main()