import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Read Data Set.
data_set = pd.read_csv("../res/length_weight.csv")

# Visualize Features
data_set["length"].hist().plot()
plt.title("Height Data")
plt.xlabel("Height (cm)")
plt.ylabel("Number of People")
plt.show()

# Features
x_train = [data_set["length"]]

# Labels
y_train = [data_set["weight"]]
# Setup variables
W = tf.Variable(np.random.rand(1, 1).astype(np.float32))
b = tf.Variable(np.random.rand(1, 1).astype(np.float32))

# Setup placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# model = xW + b
model = tf.matmul(x, W, transpose_a=True) + b

# loss = sqrt(model - y)
loss = tf.reduce_mean(tf.square(model - y))

# Model Training Operation
minimize_op = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(loss)

# Training Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(500000):
        sess.run(minimize_op, feed_dict={x: x_train, y: y_train})

    # Evaluate training accuracy
    W, b, loss = sess.run([W, b, loss], feed_dict={x: x_train, y: y_train})
    print("W = %s, b = %s, loss = %s" % (W[0][0], b[0][0], loss))

    # Visualize Result
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'o', c="blue")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    min_max = np.mat([[np.min(x_train)], [np.max(x_train)]])

    def f(x):
        return x * W + b

    ax.plot(min_max, f(min_max), c="red")
    plt.title("Linear Regression Model")
    plt.show()
