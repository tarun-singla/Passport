import numpy as np
import pickle
import csv

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

def import_data():
    X = np.genfromtxt("train_X_de.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_de.csv", delimiter=',', dtype=np.float64)
    #X = X[:int(0.8*len(X))]
    #Y = Y[:int(0.8*len(Y))]
    return X, Y

def calculate_gini_index(Y_subsets):
    gini_index = 0
    total_instances = sum(len(Y) for Y in Y_subsets)
    classes = sorted(set([j for i in Y_subsets for j in i]))

    for Y in Y_subsets:
        m = len(Y)
        if m == 0:
            continue
        count = [Y.count(c) for c in classes]
        gini = 1.0 - sum((n / m) ** 2 for n in count)
        gini_index += (m / total_instances)*gini
    
    return gini_index

def split_data_set(data_X, data_Y, feature_index, threshold):
    left_X = []
    right_X = []
    left_Y = []
    right_Y = []
    for i in range(len(data_X)):
        if data_X[i][feature_index] < threshold:
            left_X.append(data_X[i])
            left_Y.append(data_Y[i])
        else:
            right_X.append(data_X[i])
            right_Y.append(data_Y[i])
    
    return left_X, left_Y, right_X, right_Y

def get_best_split(X, Y):
    X = np.array(X)
    best_gini_index = 9999
    best_feature = 0
    best_threshold = 0
    for i in range(len(X[0])):
        thresholds = set(sorted(X[:, i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 or len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t
    return best_feature, best_threshold

def construct_tree(X, Y, max_depth, min_size, depth):
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)

    #check is pure
    if len(set(Y)) == 1:
        return node
    
    #check max depth reached
    if depth >= max_depth:
        return node

    #check min subset at node
    if len(Y) <= min_size:
        return node

    feature_index, threshold = get_best_split(X, Y)

    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold
    
    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth + 1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth + 1)
    return node

def train_model(X, Y):
    max_depth = 10
    min_size = 5
    root = construct_tree(X, Y, max_depth, min_size, 0)
    return root

def save_model(root, weights_file_name):
    with open(weights_file_name, 'wb') as weights_file:
        pickle.dump(root, weights_file, pickle.HIGHEST_PROTOCOL)
        weights_file.close()
        
if __name__ == "__main__":
    X, Y = import_data()
    root = train_model(X, Y)
    save_model(root, "MODEL_FILE.sav")
