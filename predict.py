import numpy as np
import csv
import sys
import pickle
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_de.csv".
"""

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    #test_X = test_X[int(0.8*len(test_X)):]
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model

def predict_value(root, X):
    node = root
    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class

def predict_target_values(test_X, model):
    test_Y = []
    for i in range(len(test_X)):
        k = predict_value(model, test_X[i])
        test_Y.append(k)
    return np.array(test_Y)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 
