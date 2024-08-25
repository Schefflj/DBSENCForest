import pandas as pd
from sklearn.preprocessing import LabelEncoder



class ExtDataFrame:
    def __init__(self, dataframe):
        """
        Initializes the ExtDataFrame object.

        :param dataframe: pandas DataFrame, the input DataFrame to be extended.
        """
        self.sample_dict = {}
        self.dataframe = dataframe
        for index, row in dataframe.iterrows():
            self.sample_dict[index] = Sample(index, row, dataframe)

        self.label_encoder = LabelEncoder()
        labels = self.get_labels_list()
        self.label_encoder.fit(labels)
        self.label_mapping = {label: index for index, label in enumerate(self.label_encoder.classes_)}
        self.invert_label_mapping = {v: k for k, v in self.label_mapping.items()}


    def add_data(self, data):
        """
        Adds new data to the DataFrame and updates the sample dictionary and label encoder.

        :param data: pandas DataFrame, the new data to be added.
        """
        self.dataframe = pd.concat([self.dataframe, data])
        self.dataframe.reset_index(drop=True, inplace=True)

        for index, row in self.dataframe.iterrows():
            if index not in self.sample_dict:
                self.sample_dict[index] = Sample(index, row, self.dataframe)
            else:
                self.sample_dict[index].dataframe = self.dataframe

        self.label_encoder.fit(self.get_labels_list())
        self.label_mapping = {label: index for index, label in enumerate(self.label_encoder.classes_)}
        self.invert_label_mapping = {v: k for k, v in self.label_mapping.items()}


    def get_feature_samples_numpy(self):
        """
        Extracts feature samples as a NumPy array, excluding the label column.

        :return: numpy array, feature samples converted to a NumPy array.
        """
        tmp_dataframe = self.dataframe.copy()
        tmp_dataframe.pop(tmp_dataframe.columns[-1])

        return tmp_dataframe.apply(pd.to_numeric, errors='coerce').to_numpy()


    def get_labels_list(self):
        """
        Retrieves the list of labels from the DataFrame.

        :return: list, list of labels from the last column of the DataFrame.
        """
        tmp_dataframe = self.dataframe.copy()

        return tmp_dataframe.iloc[:, -1].tolist()

    def get_encoded_labels_numpy(self):
        """
        Retrieves the labels encoded as a NumPy array.

        :return: numpy array, encoded labels.
        """
        return self.label_encoder.transform(self.get_labels_list())


class Sample():
    def __init__(self, index, row, dataframe):
        """
        Initializes a Sample object.

        :param index: int, the index of the sample in the DataFrame.
        :param row: pandas Series, the row of the sample data.
        :param dataframe: pandas DataFrame, the DataFrame the sample belongs to.
        """
        self.row = row
        self.index = index
        self.dataframe = dataframe
        self.notes = ""



