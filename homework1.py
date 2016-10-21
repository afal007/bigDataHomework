import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

iris_data = [list(elem) for elem in zip(iris.data, iris.target)]

np.random.shuffle(iris_data)

iris_data_train = iris_data[:120]
iris_data_test = iris_data[120:]

'''
    u and x looks like [characteristics]
    return: scalar
'''
def distance(u, x):
    diff = list(np.array(u) - np.array(x))
    return np.dot(np.transpose(diff), diff) ** 0.5


'''
    train_data and test_data looks like [[[characteristics], class], ... , [[characteristics], class]]
'''
def nearest_neighbour(train_data, test_data):
    distances = []
    result = []
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


print(nearest_neighbour(iris_data_train, iris_data_test))
target_test = []
for i in iris_data_test:
    target_test.append(i[1])
print(target_test)
#TODO: accuracy, presicion, polnota