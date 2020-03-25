import tensorflow as tf
import tensorflow.keras as keras

(x,y),(x_test,y_test) = keras.datasets.mnist.load_data('./mnist.npz');

print(x_test)

# from tensorflow.examples.tutorials.mnist import input_data
