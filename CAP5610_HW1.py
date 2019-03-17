from keras.datasets import mnist
import math
import numpy as np

def sigmoid(z):
  return 1 / (1 + math.exp(-z))


def softmax(z):
    pass


def sgd_mse(a, y, x, w, b, lr):
    a_prime = 1 - a
    w_gradient = ((a - y) * a_prime) * x
    b_gradient = (a - y) * a_prime
    w_update = w - lr * w_gradient
    b_update = b - lr * b_gradient
    return w_update, b_update


def sgd_bce(a, y, x, w, b, lr):
    w_gradient = (a - y) * x
    b_gradient = a - y
    w_update = w - lr * w_gradient
    b_update = b - lr * b_gradient
    return w_update, b_update


def sgd_cce(a, y, x, w, b, lr):
    pass

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train_mat = [[], [], [], [], [], [], [], [], [], []]

    for label in y_train:
        y_train_mat[label].append(1)
        for i in range(10):
            if i != label:
                y_train_mat[i].append(0)

    for i in range(10):
        y_train_mat[i] = np.array(y_train_mat[i])

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255

    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test.astype('float32') / 255

    w = [0] * 784
    b = [0]

    lr = .001

    correct = incorrect = 0

    classifiers = [0] * 10

    #Train k classifiers
    for k in range(10):
        #Train for 100 epoch
        for epoch in range(10):
            #Train with batch size of 1
            for i in range(y_train_mat[k].shape[0]):
                x = x_train[i]
                z = np.dot(x, w) + b
                a = sigmoid(z)
                if a >= .5:
                    is_digit_prediction = True
                else:
                    is_digit_prediction = False

                y = y_train_mat[k][i]

                w, b = sgd_bce(a, y, x, w, b, lr)

                if epoch > 90:
                    if (y and is_digit_prediction) or (not y and not is_digit_prediction):
                        #Correct prediction
                        correct += 1
                    else:
                        #Incorrect prediction
                        incorrect += 1
        classifiers[k] = (w,b)