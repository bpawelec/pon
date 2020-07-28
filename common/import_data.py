import pandas as pd
import numpy


class ImportData:
    def __init__(self,
                 dataset_path='../dataset/winequality-red.csv',
                 columns_path='../dataset/winequality-red.names'):
        self.dataset_path = dataset_path
        self.columns_path = columns_path

    def import_names_of_columns(self)-> numpy.ndarray:
        columns = pd.read_csv(self.columns_path, sep=';', comment='#', header=None).to_numpy()

        return numpy.concatenate(columns, axis=0)

    def import_columns(self, selected_columns_names: numpy.ndarray)-> numpy.ndarray:

        columns_names = self.import_names_of_columns()

        data = pd.read_csv(self.dataset_path, sep=';', names=columns_names, usecols=selected_columns_names)

        return data.values

    def import_columns_without_class(self) -> numpy.ndarray:
        columns_names = self.import_names_of_columns()
        result = numpy.take(columns_names, range(-1, 11))

        return result

    def get_columns_from_data(self, columns: []) -> numpy.ndarray:
        columns_names = self.import_names_of_columns()
        usecols = self.import_columns_without_class()
        data = pd.read_csv(self.dataset_path, sep=';', index_col=-1, names=columns_names, usecols=usecols)
        tmp = data.drop(columns=columns)
        return tmp.values

    def import_all_data(self) -> numpy.ndarray:

        columns_names = self.import_names_of_columns()
        usecols = self.import_columns_without_class()

        data = pd.read_csv(self.dataset_path, sep=';', index_col=-1, names=columns_names, usecols=usecols)
        return data.values

    def import_train_data(self) -> numpy.ndarray:

        columns_names = self.import_names_of_columns()
        usecols = self.import_columns_without_class()

        data = pd.read_csv(self.dataset_path, sep=';', index_col=-1, names=columns_names,
                           usecols=usecols, nrows=2000)

        return data.values

    def import_test_data(self) -> numpy.ndarray:

        columns_names = self.import_names_of_columns()
        usecols = self.import_columns_without_class()

        data = pd.read_csv(self.dataset_path, sep=';', index_col=-1, names=columns_names,
                           usecols=usecols, skiprows=2000, nrows=2000)

        return data.values

    def import_columns_train(self, selected_columns_names: numpy.ndarray)-> numpy.ndarray:

        columns_names = self.import_names_of_columns()

        data = pd.read_csv(self.dataset_path, sep=';', names=columns_names,
                           usecols=selected_columns_names, nrows=2000)

        return data.values

    def import_columns_test(self, selected_columns_names: numpy.ndarray)-> numpy.ndarray:

        columns_names = self.import_names_of_columns()

        data = pd.read_csv(self.dataset_path, sep=';', names=columns_names,
                           usecols=selected_columns_names, skiprows=2000, nrows=2000)

        return data.values
