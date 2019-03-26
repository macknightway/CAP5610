from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import math
import numpy as np
import sys


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def softmax(z, z_list):
    z_vec = np.array(z_list)
    denom = np.sum(np.exp(z_vec))
    num = np.exp(z)
    return np.divide(num,denom)


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


def sgd_cce(a, y, x, w, b, lr, j):
    gradient = 0
    w_gradient = 0
    b_gradient = 0
    for k in range(10):
        if k == j:
            k_delta = 1
        else:
            k_delta = 0
        gradient = gradient + (y[k] * (a - k_delta))
    w_gradient = gradient * x
    b_gradient = gradient
    w_update = w - lr * w_gradient
    b_update = b - lr * b_gradient
    return w_update, b_update


def preprocess_image(image):
    whitespace_regions = np.zeros((30, 30))
    img_len = img_width = 28
    for i in range(img_len):
        for j in range(img_width):
            if image[i][j] > 50:
                image[i][j] = 255
            else:
                image[i][j] = 0
                whitespace_regions[i + 1][j + 1] = 1
    np.set_printoptions(linewidth = 180)
    return image, whitespace_regions

def connected_components(image):
    image, whitespace_regions = preprocess_image(image)
    need_to_visit = whitespace_regions
    num_whitespace_regions = 1
    i = j = 1
    visit_stack = []
    print(need_to_visit)
    while np.any(need_to_visit):
        #Initial case
        if need_to_visit[i][j]:
            need_to_visit[i][j] = 0
            visit_stack.append((i,j))
        #Down-Right
        elif need_to_visit[i+1][j+1]:
            i = i + 1
            j = j + 1
            need_to_visit[i][j] = 0
            visit_stack.append((i,j))
        #Right
        elif need_to_visit[i][j+1]:
            j = j + 1
            need_to_visit[i][j] = 0
            visit_stack.append((i,j))
        #Up-Right
        elif need_to_visit[i-1][j+1]:
            i = i - 1
            j = j + 1
            need_to_visit[i][j] = 0
            visit_stack.append((i, j))
        #Up
        elif need_to_visit[i-1][j]:
            i = i - 1
            need_to_visit[i][j] = 0
            visit_stack.append((i, j))
        #Up-Left
        elif need_to_visit[i-1][j-1]:
            i = i - 1
            j = j - 1
            need_to_visit[i][j] = 0
            visit_stack.append((i, j))
        #Left
        elif need_to_visit[i][j-1]:
            j = j - 1
            need_to_visit[i][j] = 0
            visit_stack.append((i, j))
        #Down-Left
        elif need_to_visit[i+1][j-1]:
            i = i + 1
            j = j - 1
            need_to_visit[i][j] = 0
            visit_stack.append((i, j))
        #Down
        elif need_to_visit[i+1][j]:
            i = i + 1
            need_to_visit[i][j] = 0
            visit_stack.append((i, j))
        else:
            if visit_stack:
                i,j = visit_stack.pop()
            else:
                num_whitespace_regions += 1
                for k in range(30):
                    for w in range(30):
                        if need_to_visit[k][w]:
                            i,j = k,w
    return num_whitespace_regions


def logistic_regression_keras():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255

    model = Sequential()
    model.add(Dense(784, input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=1)

def logistic_regression_softmax():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255

    #x_test = x_test.reshape((10000, 28 * 28))
    #x_test = x_test.astype('float32') / 255

    w = [[0] * 784, [0] * 784,[0] * 784,[0] * 784,[0] * 784,
         [0] * 784, [0] * 784,[0] * 784,[0] * 784,[0] * 784]
    b = [0] * 10

    lr = .001

    for epoch in range(10):
        # Train with batch size of 1
        for i in range(y_train.shape[0]):
            z = []
            a = []
            x = x_train[i]
            for neuron in range(10):
                z.append(np.dot(x, w[neuron]))
            for neuron in range(10):
                a.append(softmax(z[neuron], z))

            if i == 45:
                print(a)

            y_hat = np.argmax(a)
            y_arg_max = np.argmax(y_train[i])

            if y_hat == y_arg_max:
                is_digit_prediction = True
            else:
                is_digit_prediction = False

            for j in range(10):
                w[j], b[j] = sgd_cce(a[j], y_train[i], x, w[j], b[j], lr, j)

            if i % 1000 == 0:
                print("Y_Hat: {0}".format(y_hat))
                print("Y: {0}".format(y_arg_max))
                print(a)
                print("")



def logistic_regression_sigmoid():
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

    #x_test = x_test.reshape((10000, 28 * 28))
    #x_test = x_test.astype('float32') / 255

    w = [0] * 784
    b = [0]

    lr = .001

    correct = incorrect = 0

    classifiers = [0] * 10

    # Train k classifiers
    for k in range(10):
        # Train for 100 epoch
        for epoch in range(10):
            # Train with batch size of 1
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
                        # Correct prediction
                        correct += 1
                    else:
                        # Incorrect prediction
                        incorrect += 1
        classifiers[k] = (w, b)

if __name__ == '__main__':
    #logistic_regression_sigmoid()
    #logistic_regression_softmax()
    #logistic_regression_keras()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)