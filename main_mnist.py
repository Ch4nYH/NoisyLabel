import tensorflow.compat.v1 as tf
import numpy as np
from absl import app
from absl import flags
from absl import logging

import os

flags.DEFINE_string(
        "tpu", default=None,
        help="The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")
flags.DEFINE_string(
        "tpu_zone", default=None,
        help="[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")
flags.DEFINE_string(
        "gcp_project", default=None,
        help="[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

# Model specific parameters
flags.DEFINE_string("data_dir", "",
                                        "Path to directory containing the MNIST dataset")
flags.DEFINE_string("model_dir", None, "Estimator model_dir")
flags.DEFINE_integer("batch_size", 1024,
                                         "Mini-batch size for the training. Note that this "
                                         "is the global batch size and not the per-shard batch.")
flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
flags.DEFINE_integer("eval_steps", 0,
                                         "Total number of evaluation steps. If `0`, evaluation "
                                         "after training is skipped.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
flags.DEFINE_integer("iterations", 50,
                                         "Number of iterations per TPU training loop.")
flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = flags.FLAGS
num_classes = 10


def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}

def model_fn(features, labels, mode, params):
    image = features / 255.
    act = tf.nn.leaky_relu
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True
    else:
        training = False 
    conv1 = tf.layers.Conv2D(filters = 32,
                         kernel_size = 3,
                         padding = "same")
    conv2 = tf.layers.Conv2D(filters = 128,
                        kernel_size = 3,
                        padding = "same")
    conv3 = tf.layers.Conv2D(filters = 128,
                        kernel_size = 3,
                        padding = "same")
    conv4 = tf.layers.Conv2D(filters = 256,
                        kernel_size = 3,
                        padding = "same")
    conv5 = tf.layers.Conv2D(filters = 256,
                        kernel_size = 3,
                        padding = "same")
    conv6 = tf.layers.Conv2D(filters = 256,
                        kernel_size = 3,
                        padding = "same")
    conv7 = tf.layers.Conv2D(filters = 512,
                        kernel_size = 3,
                        padding = "same")
    conv8 = tf.layers.Conv2D(filters = 256,
                        kernel_size = 3,
                        padding = "same")
    conv9 = tf.layers.Conv2D(filters = 128,
                        kernel_size = 3,
                        padding = "same")
    bn1 = tf.layers.BatchNormalization()
    bn2 = tf.layers.BatchNormalization()
    bn3 = tf.layers.BatchNormalization()
    bn4 = tf.layers.BatchNormalization()
    bn5 = tf.layers.BatchNormalization()
    bn6 = tf.layers.BatchNormalization()
    bn7 = tf.layers.BatchNormalization()
    bn8 = tf.layers.BatchNormalization()
    bn9 = tf.layers.BatchNormalization()

    f1 = act(bn1(conv1(features), training = training), alpha = 0.1)
    f2 = act(bn2(conv2(f1), training = training), alpha = 0.1)
    f3 = act(bn3(conv3(f2), training = training), alpha = 0.1)
    f3 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(f3)
    f3 = tf.layers.Dropout(0.25)(f3, training=(mode == tf.estimator.ModeKeys.TRAIN))
    f4 = act(bn4(conv4(f3), training = training), alpha = 0.1)
    f5 = act(bn5(conv5(f4), training = training), alpha = 0.1)
    f6 = act(bn6(conv6(f5), training = training), alpha = 0.1)
    f6 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(f6)
    f6 = tf.layers.Dropout(0.25)(f6, training=(mode == tf.estimator.ModeKeys.TRAIN))
    f7 = act(bn7(conv7(f6), training = training), alpha = 0.1)
    f8 = act(bn8(conv8(f7), training = training), alpha = 0.1)
    f9 = act(bn9(conv9(f8), training = training), alpha = 0.1)

    shape = f9.get_shape()
    inputSize = np.array([shape[1],shape[2]]).astype(np.int32)
    outputSize = np.array([1,1])
    strideSize = np.floor(inputSize//outputSize).astype(np.int32)

    kernelSize = inputSize - (outputSize-1) * strideSize

    flatten = tf.layers.Flatten()(tf.layers.AveragePooling2D(pool_size=kernelSize,
                                                         strides=strideSize,
                                                         padding="same")(f9))
    logits = tf.layers.Dense(num_classes)(flatten)
    predictions = {
                "class_ids": tf.argmax(logits, axis=1),
                "probabilities": tf.nn.softmax(logits),
        }
    accuracy = tf.metrics.accuracy(labels=labels,
            predictions=predictions['class_ids'])
    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        
        return tf.estimator.tpu.TPUEstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar("accuracy", accuracy[1])
    tf.summary.scalar("loss", loss)
    summary_hook = tf.train.SummarySaverHook(
        500,
        output_dir='log/',
        summary_op=tf.summary.merge_all())
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                decay_steps=100000,
                decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        if FLAGS.use_tpu:
            optimizer = tf.tpu.CrossShardOptimizer(optimizer)
        
        return tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_global_step()),
                training_hooks = [summary_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


def dataset(records_file):
    """Loads MNIST dataset from given TFRecords file."""
    features = {
            "image_raw": tf.io.FixedLenFeature((), tf.string),
            "label": tf.io.FixedLenFeature((), tf.int64),
    }

    def decode_record(record):
        example = tf.io.parse_single_example(record, features)
        image = tf.decode_raw(example["image_raw"], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [28, 28, 1])

        return image, example["label"]

    return tf.data.TFRecordDataset(records_file).map(decode_record)


def train_input_fn(params):
    """train_input_fn defines the input pipeline used for training."""
    batch_size = params["batch_size"]
    records_file = os.path.join(params["data_dir"], "train.tfrecords")

    return dataset(records_file).cache().repeat().shuffle(
            buffer_size=50000).batch(batch_size, drop_remainder=True)


def eval_input_fn(params):
    batch_size = params["batch_size"]
    records_file = os.path.join(params["data_dir"], "validation.tfrecords")

    return dataset(records_file).batch(batch_size, drop_remainder=True)


def predict_input_fn(params):
    batch_size = params["batch_size"]
    records_file = os.path.join(params["data_dir"], "test.tfrecords")

    # Take out top 10 samples from test data to make the predictions.
    return dataset(records_file).take(100).batch(batch_size)

def main(argv):
    del argv  # Unused.
    logging.set_verbosity(logging.INFO)

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu if (FLAGS.tpu or FLAGS.use_tpu) else "",
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project
    )

    run_config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            session_config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.estimator.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
    )

    estimator = tf.estimator.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=FLAGS.use_tpu,
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size,
            predict_batch_size=FLAGS.batch_size,
            params={"data_dir": FLAGS.data_dir},
            config=run_config)
    # TPUEstimator.train *requires* a max_steps argument.
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
    # TPUEstimator.evaluate *requires* a steps argument.
    # Note that the number of examples used during evaluation is
    # --eval_steps * --batch_size.
    # So if you change --batch_size then change --eval_steps too.
    if FLAGS.eval_steps:
        loss, eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)
        print(eval_metrics)
    # Run prediction on top few samples of test data.
    if FLAGS.enable_predict:
        predictions = estimator.predict(input_fn=predict_input_fn)

        for pred_dict in predictions:
            template = ('Prediction is "{}" ({:.1f}%).')

            class_id = pred_dict["class_ids"]
            probability = pred_dict["probabilities"][class_id]

            logging.info(template.format(class_id, 100 * probability))


if __name__ == "__main__":
    app.run(main)
