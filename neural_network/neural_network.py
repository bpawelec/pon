import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from common.import_data import ImportData

if __name__ == "__main__":
    data_set = ImportData()

    x_train, x_test, y_train, y_test = \
        train_test_split(data_set.import_all_data(),
                         data_set.import_columns
                         (np.array(['quality'])),
                         test_size=0.2, random_state=13)

    NN = MLPClassifier(solver='adam', alpha=0.0001,
                       hidden_layer_sizes=(21, 3),
                       random_state=1, max_iter=2000, verbose=1).fit(x_train, y_train.ravel())
    predictions = NN.predict(x_train)
    print(predictions)
    print(round(NN.score(x_test, y_test.ravel()), 4))
















