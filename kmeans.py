import csv
import numpy as np

PATH = "./data/spendinglevel.csv"

with open(PATH, newline='') as file:
    data_csv = csv.reader(file)
    data = np.array(list(data_csv))


feature = data[1:, 1:3].astype(float)
centroid = feature[:3]


def match(data_point):
    value_point = []
    for element in data_point:
        value_point.append(data_point[element])
    return np.array(value_point).reshape(-1, 2)


def kmeans_regression(table, k):

    feature = table[1:, 1:3].astype(float)
    centroid = feature[2:5]

    for _ in range(100):
        distance = np.array(
            [np.sqrt(np.sum((feature - centroid[i, :])**2, axis=1)) for i in range(k)]).T
        votes = np.argmin(distance, axis=1)
        new_centroids = np.array(
            [feature[votes == i].mean(axis=0) for i in range(k)])

        if np.all(new_centroids == centroid):
            break
        else:
            centroid = new_centroids

    return centroid


def type_of_age(centroid_data, new_dt, label):
    new_age = match(new_dt)
    new_distance = np.array(
        [np.sqrt(np.sum((centroid_data - new_age)**2, axis=0))])
    # print(centroid_data.shape, new_age.shape)
    new_label = np.argmin(new_distance)

    for element in label:
        if new_label == element:
            return label[element]

    return None


if __name__ == '__main__':
    point = {
        'age': 24,
        'expenditure': 78
    }

    label = {
        0: 'younger',
        1: 'middle-aged',
        2: 'old'
    }
    centroid = kmeans_regression(data, 3).round(2)

    print("we have 3 centroids:")
    for x in range(1, 4):
        print("centroid", x, ":", tuple(centroid[x-1, :]), end='\n')
    result = type_of_age(centroid, point, label)

    print("\nlabel of point: ", result)
