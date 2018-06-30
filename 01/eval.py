from net import *
import pickle
from sklearn.metrics import log_loss
import numpy as np

def show_max_prob(image, label, net):
    probabilities = net.forward(np.append(image, 1))
    digit = np.argmax(probabilities.numpy())
    print("Digit number {} considered as {}".format(label, digit))
    return probabilities

check_point = input("Enter checkpoint name: ")
f = open('model_save/'+check_point, 'rb')
net = pickle.load(f)

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets("MNIST")

res = []
ans = []
for i in range(len(mnist.test.images)):
    res.append(list(show_max_prob(mnist.test.images[i], mnist.test.labels[i], net)))
    ans.append(mnist.test.labels[i])
    
print("LOGLOSS: ", log_loss(ans, res))
print("ACCURACY: ", sum([np.argmax(x) == y for x, y in zip(res, ans)]) / len(ans))
