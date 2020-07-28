import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from common.import_data import ImportData



if __name__ == "__main__":

    data_set = ImportData()
x: np.ndarray = data_set.import_train_data()
y: np.ndarray = data_set.import_columns_train(
        np.array(['quality']))
name_of_columns: np.ndarray = data_set.import_names_of_columns()
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=['fixed acidity', 'volatile acidity', 'citric acid',
                                                                'residual sugar', 'chlorides', 'free sulfur dioxide',
                                                                'total sulfur dioxide', 'density', 'ph', 'sulphates',
                                                                'alcohol'])
feat_importances.nlargest(11).plot(kind='barh')
plt.xlabel("Znaczenie cech")
plt.show()