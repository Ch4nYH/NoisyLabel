import tensorflow as tf
from tensorflow import keras

class WeightedCrossEntropy(keras.losses.Loss):
    def __init__(self, name='weighted_crossentropy'):
        
        super().__init__(reduction = keras.losses.Reduction.AUTO, 
                         name = 'weighed_crossentropy')
        self.from_logits = False
        
    def call(self, y_true, y_pred, w):
        ce = tf.losses.categorical_crossentropy(
            y_true, y_pred, from_logits = self.from_logits
        )[:, None]
        ce = w * ce
        return ce
        
        