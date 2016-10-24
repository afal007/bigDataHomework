import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

iris_data = [list(elem) for elem in zip(iris.data, iris.target)]

np.random.shuffle(iris_data)

iris_data_train = iris_data[:20]
iris_data_test = iris_data[20:]

'''
    distance(vector, vector)

    u and x looks like [characteristics]
    return: scalar
'''


def distance(u, x):
    diff = list(np.array(u) - np.array(x))
    return np.dot(np.transpose(diff), diff) ** 0.5


'''
    nearest_neighbour(train_dataset, test_dataset)

    train_data and test_data looks like [[[characteristics], class], ... , [[characteristics], class]]
'''


def nearest_neighbour(train_data, test_data):
    distances, result = [], []

    for u in test_data:
        for j, x in enumerate(train_data):
            distances.append([distance(u[0], x[0]), j])  # list of [distance, element_number]
        distances.sort(key=lambda t: t[0])
        '''
            result - list of classes.
            take element number distances[0][1] from train_data and return its class.
        '''
        result.append(train_data[distances[0][1]][1])
        distances.clear()
    return result


'''
    eval_class(distances, dataset, neighbours_num)

    fills classes vector.
    train_data[n][1] - class number
    dist[i][1] - elem number
    return: index of max elem in classes
'''


def eval_class(dist, train_data, k):
    classes = [0, 0, 0]
    for i in range(k):
        classes[train_data[dist[i][1]][1]] += 1
    return classes.index(max(classes))


def k_nearest_neighbours(train_data, test_data, k):
    distances, result = [], []

    for u in test_data:
        for j, x in enumerate(train_data):
            distances.append([distance(u[0], x[0]), j])  # list of [distance, element_number]
        distances.sort(key=lambda t: t[0])

        result.append(eval_class(distances, train_data, k))
        distances.clear()
    return result


'''
    eval_class_weighed(distances, dataset, neighbours_num, weight_base)

    exponential weight.
    fills classes vector.
    train_data[n][1] - class number
    dist[i][1] - elem number
    return: index of max elem in classes
'''


def eval_class_weighed(dist, train_data, k, q):
    classes = [0, 0, 0]
    for i in range(k):
        classes[train_data[dist[i][1]][1]] += q ** i
    return classes.index(max(classes))


def weighed_k_nearest_neighbours(train_data, test_data, k, q):
    distances, result = [], []

    for u in test_data:
        for j, x in enumerate(train_data):
            distances.append([distance(u[0], x[0]), j])  # list of [distance, element_number]
        distances.sort(key=lambda t: t[0])

        result.append(eval_class_weighed(distances, train_data, k, q))
        distances.clear()
    return result


def kernel(x):
    return (3 / 4) * (1 - x ** 2) * (x <= 1)

def eval_class_kernel(dist, train_data, k):
    classes = [0, 0, 0]
    for i in range(k):
        classes[train_data[dist[i][1]][1]] += kernel(dist[i][0]/dist[k][0])
    return classes.index(max(classes))


def parzen_window(train_data, test_data, k):
    distances, result = [], []

    for u in test_data:
        for j, x in enumerate(train_data):
            distances.append([distance(u[0], x[0]), j])  # list of [distance, element_number]
        distances.sort(key=lambda t: t[0])

        result.append(eval_class_kernel(distances, train_data, k))
        distances.clear()
    return result


def potential_k(x):
    return 1/(x+1)


def eval_class_potential(dist, train_data, gamma):
    classes = [0, 0, 0]

    for i in range(len(train_data)):
        classes[train_data[dist[i][1]][1]] += gamma[i] * potential_k(dist[i][0]/5)
    return classes.index(max(classes))

def eval_error(result, target):
    err = 0
    for elem in zip(result, target):
        if elem[0] != elem[1]:
            err += 1
    return err

def tune_potential(train_data):
    distances, result = [], []
    gamma = [0 for i in range(len(train_data))]
    error = 7

    while error > 0:
        result.clear()
        for i, u in enumerate(train_data):
            for j, x in enumerate(train_data):
                distances.append([distance(u[0], x[0]), j])  # list of [distance, element_number]
            distances.sort(key=lambda t: t[0])
            distances[0][0] = 99999

            result.append(eval_class_potential(distances, train_data, gamma))
            distances.clear()
            if result[i] != u[1]:
                gamma[i] += 1
        error = eval_error(result, [x[1] in train_data])
    return gamma

def potential_funcs(train_data, test_data):
    distances, result = [], []
    gamma = tune_potential(train_data)
    print("\nGamma: ", gamma)

    for u in test_data:
        for j, x in enumerate(train_data):
            distances.append([distance(u[0], x[0]), j])  # list of [distance, element_number]
        distances.sort(key=lambda t: t[0])

        result.append(eval_class_potential(distances, train_data, gamma))
        distances.clear()
    return result

def accuracy(result, target):
    positive, negative = 0, 0
    for elem in zip(result, target):
        if elem[0] == elem[1]:
            positive += 1
        else:
            negative += 1
    return positive / (negative + positive)


def fill_accuracy_matrix(result, target):
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for elem in zip(result, target):
        matrix[elem[0]][elem[1]] += 1
    return matrix

target_test = []
for i in iris_data_test:
    target_test.append(i[1])
print("Target: ")
print(target_test)

result = nearest_neighbour(iris_data_train, iris_data_test)
print("\nNearest neighbour: ")
print(result)
print(accuracy(result, target_test))
print(fill_accuracy_matrix(result, target_test))

result = k_nearest_neighbours(iris_data_train, iris_data_test, 4)
print("\nk nearest neighbours: ")
print(result)
print(accuracy(result, target_test))
print(fill_accuracy_matrix(result, target_test))

result = weighed_k_nearest_neighbours(iris_data_train, iris_data_test, 3, 0.6)
print("\nWeighed k nearest neighbours: ")
print(result)
print(accuracy(result, target_test))
print(fill_accuracy_matrix(result, target_test))

result = parzen_window(iris_data_train, iris_data_test, 5)
print("\nParzen window: ")
print(result)
print(accuracy(result, target_test))
print(fill_accuracy_matrix(result, target_test))

result = potential_funcs(iris_data_train, iris_data_test)
print("\nPotential funcs: ")
print(result)
print(accuracy(result, target_test))
print(fill_accuracy_matrix(result, target_test))