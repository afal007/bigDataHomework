import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import scipy.integrate as integrate
import math

from sklearn.datasets import load_iris

iris = load_iris()

iris_data = [list(elem) for elem in zip(iris.data, iris.target)]

np.random.shuffle(iris_data)


class LogisticRegression(object):
    def __init__(self, data, class_num, treshold):
        self.data = data
        self.treshold = treshold
        self.class_num = class_num
        self.result = []
        self.theta = []

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def apply(self):
        for i in range(len(self.data)):
            res = self.sigmoid(np.dot(np.transpose(self.theta), self.data[i][0]))
            self.result.append([res, res > self.treshold])

    def diff(self, x, u):
        diff = list(np.array(u) - np.array(x))
        return np.dot(np.transpose(diff), diff) ** 0.5

    def train(self, alfa):
        param_arr = [self.data[i][0] for i in range(len(self.data))]
        target = [self.data[i][1] for i in range(len(self.data))]

        train_data = [elem for elem in zip(param_arr, [x == self.class_num for x in target])]

        self.theta = [-1] * len(self.data[0][0])
        theta_next = [1] * len(self.data[0][0])
        epsilon = 0.001

        while self.diff(theta_next, self.theta) > epsilon:
            self.theta = theta_next[:]

            for i in range(len(self.theta)):
                summ = 0
                for j in range(len(train_data)):
                    summ += (self.sigmoid(np.dot(np.transpose(self.theta), self.data[j][0])) - train_data[j][1]) * \
                           train_data[j][0][i]

                theta_next[i] = self.theta[i] - (alfa * summ) / len(train_data)

    def check(self):
        tp = fp = fn = 0
        for i in range(len(self.data)):
            if self.result[i][1] == 1 and self.data[i][1] == self.class_num:
                tp += 1
            elif self.result[i][1] == 1 and self.data[i][1] != self.class_num:
                fp += 1
            elif self.result[i][1] == 0 and self.data[i][1] == self.class_num:
                fn += 1
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        print("\nRecall = ", recall, " Precision = ", precision)

    def print_roc(self):
        positive = negative = 0
        for i in range(len(self.data)):
            if self.result[i][1] == 1:
                positive += 1
            else:
                negative += 1
        self.result.sort(key=lambda t: t[0])

        _x = [0]
        _y = [0]
        auc = 0
        for i in range(1, len(self.data)):
            if self.result[i][1] == 0:
                _x.append(_x[i - 1] + 1 / 100)
                _y.append(_y[i - 1])
                auc += _y[i] * (1 / 100)
            else:
                _x.append(_x[i - 1])
                _y.append(_y[i - 1] + 1 / 50)

        trace = go.Scatter(x=_x, y=_y, mode='lines+markers')
        data = [trace]

        print("AUC: ", auc)
        py.plot(data)

    def print_auc(self, x, y):
        auc = 0
        x = list(reversed(x))
        y = list(reversed(y))

        for i in range(len(x) - 1):
            auc += (y[i+1] + y[i]) * (x[i+1] - x[i])/2
        print("AUC: ", auc * 10 ** -4)

        auc = integrate.trapz(y, x)
        print("AUC lib: ", auc * 10 ** -4)

        return auc

    def print_roc_(self):
        _x = []
        _y = []
        tr = 0
        dx = 0.01
        positive = 50
        negative = 100

        while tr < 1:
            tp = fp = 0

            for i in range(len(self.result)):
                if self.result[i][0] >= tr:
                    if self.data[i][1] == self.class_num:
                        tp += 1
                    else:
                        fp += 1

            tpr = (tp/positive) * 100
            fpr = (fp/negative) * 100
            _x.append(fpr)
            _y.append(tpr)
            tr += dx

        trace = go.Scatter(x=_x, y=_y, mode='lines+markers')
        data = [trace]

        self.print_auc(_x, _y)

        py.plot(data)

    def print_result(self):
        target = [self.data[i][1] for i in range(len(self.data))]
        res = [self.result[i][1] for i in range(len(self.result))]
        res2 = [self.result[i][0] for i in range(len(self.result))]
        print(target)
        print(res)
        print(["%0.2f" % i for i in res2])

alg = LogisticRegression(iris_data, 0, 0.9)
alg.train(0.1)
alg.apply()
alg.check()
alg.print_result()
alg.print_roc_()
#alg.print_roc()

