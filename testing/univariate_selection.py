import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from common.import_data import ImportData



if __name__ == "__main__":

    data_set = ImportData()
x: np.ndarray = data_set.import_train_data()
y: np.ndarray = data_set.import_columns_train(
        np.array(['quality']))
name_of_columns: np.ndarray = data_set.import_names_of_columns()
bestfeatures = SelectKBest(score_func=chi2, k=11)
fit = bestfeatures.fit(x, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(name_of_columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(11,'Score'))  #print 10 best features
