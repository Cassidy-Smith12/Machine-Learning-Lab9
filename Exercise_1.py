import csv
import numpy as np

data = []
labels = []
with open('balloons_2features.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) 
    for row in reader:
        data.append(row[:-1])
        labels.append(row[-1])

data = np.array(data)
labels = np.array(labels)

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(set(y)) == 1:
            return y[0]
        
        if len(X[0]) == 0:
            return max(set(y), key=list(y).count)
        
        best_feature = self._best_feature_to_split(X, y)
        tree = {best_feature: {}}
        
        feature_values = set(X[:, best_feature])
        for value in feature_values:
            sub_X, sub_y = self._split_dataset(X, y, best_feature, value)
            subtree = self._build_tree(sub_X, sub_y)
            tree[best_feature][value] = subtree
        
        return tree

    def _best_feature_to_split(self, X, y):
        num_features = len(X[0])
        base_entropy = self._entropy(y)
        best_info_gain = 0
        best_feature = -1
        
        for i in range(num_features):
            feature_values = set(X[:, i])
            new_entropy = 0
            for value in feature_values:
                sub_X, sub_y = self._split_dataset(X, y, i, value)
                prob = len(sub_y) / float(len(y))
                new_entropy += prob * self._entropy(sub_y)
            
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        
        return best_feature

    def _split_dataset(self, X, y, feature, value):
        sub_X = []
        sub_y = []
        
        for i in range(len(X)):
            if X[i][feature] == value:
                reduced_X = np.concatenate((X[i][:feature], X[i][feature+1:]))
                sub_X.append(reduced_X)
                sub_y.append(y[i])
        
        return np.array(sub_X), np.array(sub_y)

    def _entropy(self, y):
        from math import log2
        num_entries = len(y)
        label_counts = {}
        
        for label in y:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        entropy = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            entropy -= prob * log2(prob)
        
        return entropy

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        return predictions

    def _predict_single(self, x):
        tree = self.tree
        while isinstance(tree, dict):
            feature_index = list(tree.keys())[0]
            feature_value = x[feature_index]
            if feature_value in tree[feature_index]:
                tree = tree[feature_index][feature_value]
            else:
                return None
        return tree

clf = DecisionTreeClassifier()
clf.fit(data, labels)

data_point = ['Stretch', 'Adult']
prediction = clf.predict([data_point])

print(f"The prediction for the data point {data_point} is: {prediction[0]}")






