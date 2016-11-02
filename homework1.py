import numpy as np
import random as rand
from sklearn.datasets import load_iris

iris = load_iris()

iris_data = [list(elem) for elem in zip(iris.data, iris.target)]

np.random.shuffle(iris_data)

iris_data_train = iris_data[:100]
iris_data_test = iris_data[100:]


class ClassificationAlgorithm(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.accuracy_matrix = []
        self.result = []
        self.target = []
        self.fill_target()

    '''
        distance(vector, vector)

        u and x looks like [characteristics]
        return: scalar
    '''

    def distance(self, u, x):
        diff = list(np.array(u) - np.array(x))
        return np.dot(np.transpose(diff), diff) ** 0.5

    def apply(self):
        distances = []

        for u in self.test_data:
            for j, x in enumerate(self.train_data):
                distances.append([self.distance(u[0], x[0]), j])  # list of [distance, element_number]
            distances.sort(key=lambda t: t[0])

            self.eval_class(distances)
            distances.clear()
        self.fill_accuracy_matrix()

    def eval_class(self, distances):
        raise NotImplementedError()

    def fill_target(self):
        for i in self.test_data:
            self.target.append(i[1])

    def get_accuracy(self):
        positive, negative = 0, 0
        for elem in zip(self.result, self.target):
            if elem[0] == elem[1]:
                positive += 1
            else:
                negative += 1
        return positive / (negative + positive)

    def fill_accuracy_matrix(self):
        self.accuracy_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for elem in zip(self.result, self.target):
            self.accuracy_matrix[elem[0]][elem[1]] += 1

    def get_precision_and_recall(self):
        result = []
        for i in range(len(self.accuracy_matrix)):
            tp = self.accuracy_matrix[i][i]
            fp, fn = 0, 0
            for j in range(len(self.accuracy_matrix)):
                if i != j:
                    fp += self.accuracy_matrix[i][j]
                    fn += self.accuracy_matrix[j][i]
            if tp + fp != 0:
                precision = tp / (tp + fp)
            else:
                precision = 0

            if tp + fn != 0:
                recall = tp / (tp + fn)
            else:
                recall = 0

            result.append(
                {"Precision": precision,
                 "Recall": recall
                 })
        return result


class NearestNeighbour(ClassificationAlgorithm):
    def eval_class(self, distances):
        self.result.append(self.train_data[distances[0][1]][1])


class KNearestNeighbours(ClassificationAlgorithm):
    def __init__(self, train_data, test_data, k=2):
        super().__init__(train_data, test_data)
        self.k = k

    def eval_class(self, distances):
        classes = [0, 0, 0]
        for i in range(self.k):
            classes[self.train_data[distances[i][1]][1]] += 1
        self.result.append(classes.index(max(classes)))


class WeighedKNearestNeighbours(ClassificationAlgorithm):
    def __init__(self, train_data, test_data, k=2, q=0.5):
        super().__init__(train_data, test_data)
        self.k = k
        self.q = q

    def eval_class(self, distances):
        classes = [0, 0, 0]
        for i in range(self.k):
            classes[self.train_data[distances[i][1]][1]] += self.q ** i
        self.result.append(classes.index(max(classes)))


class ParzenWindow(ClassificationAlgorithm):
    def __init__(self, train_data, test_data, k=2, kernel=0):
        super().__init__(train_data, test_data)
        self.k = k

        if kernel != 0:
            self.kernel = kernel
        else:
            self.kernel = self.default_kernel

    def default_kernel(self, x):
        return (3 / 4) * (1 - x ** 2) * (x <= 1)

    def eval_class(self, distances):
        classes = [0, 0, 0]
        for i in range(self.k):
            classes[self.train_data[distances[i][1]][1]] += self.kernel(distances[i][0] / distances[self.k][0])

        self.result.append(classes.index(max(classes)))


class PotentialFuncs(ClassificationAlgorithm):
    def __init__(self, train_data, test_data, h):
        super().__init__(train_data, test_data)
        self.h = h
        self.gamma = self.tune_gamma()

    def potential(self, x):
        return 1 / (x + 1)

    def tune_gamma(self):
        distances, tuned = [], []
        self.gamma = [0 for i in range(len(self.train_data))]
        error = len(self.train_data)
        cur_result = 0

        while error > 20:
            i = rand.randint(0, len(self.train_data) - 1)  # Choose random elem from train dataset
            if i in tuned:  # If gamma for this element is tuned - continue
                continue

            # Eval distances
            for j, x in enumerate(self.train_data):
                distances.append([self.distance(self.train_data[i][0], x[0]), j])  # list of [distance, element_number]
            distances.sort(key=lambda t: t[0])
            distances[0][0] = 99999

            # Apply algorithm
            cur_result = self.eval_class_gamma(distances)
            distances.clear()

            if cur_result != self.train_data[i][1]:
                # If not correct answer gamma++
                self.gamma[i] += 1
            else:
                # Else remember that we tuned this gamma and error--
                tuned.append(i)
                error -= 1
        return self.gamma

    def eval_class_gamma(self, distances):
        classes = [0, 0, 0]

        for i in range(1, self.h + 1):
            classes[self.train_data[distances[i][1]][1]] += self.gamma[i] * self.potential(distances[i][0] / distances[self.h + 1][0])
        return classes.index(max(classes))

    def eval_class(self, distances):
        classes = [0, 0, 0]

        for i in range(self.h):
            classes[self.train_data[distances[i][1]][1]] += self.gamma[i] * self.potential(distances[i][0] / distances[self.h][0])
        self.result.append(classes.index(max(classes)))

print("\nNearest neighbour: ")
alg = NearestNeighbour(iris_data_train, iris_data_test)
alg.apply()
print(alg.get_accuracy())
print(alg.accuracy_matrix)
print(alg.get_precision_and_recall())

print("\nK nearest neighbours: ")
alg = KNearestNeighbours(iris_data_train, iris_data_test, 5)
alg.apply()
print(alg.get_accuracy())
print(alg.accuracy_matrix)
print(alg.get_precision_and_recall())

print("\nWeighed k neighbours: ")
alg = WeighedKNearestNeighbours(iris_data_train, iris_data_test, 5, 0.6)
alg.apply()
print(alg.get_accuracy())
print(alg.accuracy_matrix)
print(alg.get_precision_and_recall())

print("\nParzen window: ")
alg = ParzenWindow(iris_data_train, iris_data_test, 5)
alg.apply()
print(alg.get_accuracy())
print(alg.accuracy_matrix)
print(alg.get_precision_and_recall())

print("\nPotential funcs: ")
alg = PotentialFuncs(iris_data_train, iris_data_test, 10)
alg.apply()
print(alg.get_accuracy())
print(alg.accuracy_matrix)
print(alg.get_precision_and_recall())
