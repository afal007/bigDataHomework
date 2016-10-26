import numpy as np
import random as rand
from sklearn.datasets import load_iris

iris = load_iris()

iris_data = [list(elem) for elem in zip(iris.data, iris.target)]

np.random.shuffle(iris_data)

iris_data_train = iris_data[:100]
iris_data_test = iris_data[100:]

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
        classes[train_data[dist[i][1]][1]] += kernel(dist[i][0] / dist[k][0])
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
    return 1 / (x + 1)


def eval_class_potential(dist, train_data, gamma):
    classes = [0, 0, 0]

    for i in range(1, len(train_data)):
        classes[train_data[dist[i][1]][1]] += gamma[i] * potential_k(dist[i][0] / dist[5][0])
    return classes.index(max(classes))


def tune_potential(train_data):
    distances, tuned = [], []
    gamma = [0 for i in range(len(train_data))]
    error = len(train_data)
    cur_result = 0

    while error > 20:
        i = rand.randint(0, len(train_data) - 1)  # Choose random elem from train dataset
        if i in tuned:  # If gamma for this element is tuned - continue
            continue

        # Eval distances
        for j, x in enumerate(train_data):
            distances.append([distance(train_data[i][0], x[0]), j])  # list of [distance, element_number]
        distances.sort(key=lambda t: t[0])
        distances[0][0] = 99999

        # Apply algorithm
        cur_result = eval_class_potential(distances, train_data, gamma)
        distances.clear()

        if cur_result != train_data[i][1]:
            # If not correct answer gamma++
            gamma[i] += 1
        else:
            # Else remember that we tuned this gamma and error--
            tuned.append(i)
            error -= 1
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


def precision_and_recall(matrix):
    result = []
    for i in range(len(matrix)):
        tp = matrix[i][i]
        fp, fn = 0, 0
        for j in range(len(matrix)):
            if i != j:
                fp += matrix[i][j]
                fn += matrix[j][i]
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
accuracy_matrix = fill_accuracy_matrix(result, target_test)
print(accuracy_matrix)
print(precision_and_recall(accuracy_matrix))

result = k_nearest_neighbours(iris_data_train, iris_data_test, 4)
print("\nk nearest neighbours: ")
print(result)
print(accuracy(result, target_test))
accuracy_matrix = fill_accuracy_matrix(result, target_test)
print(accuracy_matrix)
print(precision_and_recall(accuracy_matrix))

result = weighed_k_nearest_neighbours(iris_data_train, iris_data_test, 3, 0.6)
print("\nWeighed k nearest neighbours: ")
print(result)
print(accuracy(result, target_test))
accuracy_matrix = fill_accuracy_matrix(result, target_test)
print(accuracy_matrix)
print(precision_and_recall(accuracy_matrix))

result = parzen_window(iris_data_train, iris_data_test, 5)
print("\nParzen window: ")
print(result)
print(accuracy(result, target_test))
accuracy_matrix = fill_accuracy_matrix(result, target_test)
print(accuracy_matrix)
print(precision_and_recall(accuracy_matrix))

result = potential_funcs(iris_data_train, iris_data_test)
print("\nPotential funcs: ")
print(result)
print(accuracy(result, target_test))
accuracy_matrix = fill_accuracy_matrix(result, target_test)
print(accuracy_matrix)
print(precision_and_recall(accuracy_matrix))
