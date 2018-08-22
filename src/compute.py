import tensorflow as tf
import numpy as np


x_train = np.mat([[0.0] for i in range(0, 1000)])
y_train = np.mat([[0.0] for j in range(0, 1000)])


filename_queue = tf.train.string_input_producer(["../res/length_weight.csv"])
reader = tf.TextLineReader(1)
key, value = reader.read(filename_queue)

record_defaults = [[1.0], [1.0]]
col1, col2 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1])


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(0, 1000):
        example, label = sess.run([features, col2])
        x_train[i][0] = example
        y_train[i][0] = label

    coord.request_stop()
    coord.join(threads)


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Uses Mean Squared Error, although instead of mean, sum is used.
        self.loss = tf.reduce_sum(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(100000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()
