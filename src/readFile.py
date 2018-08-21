import tensorflow as tf

filenames = ["../res/length_weight.csv"]
record_defaults = [tf.float32] * 2
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True)

iterator = dataset.make_one_shot_iterator()
for x in iterator:
    print(x)