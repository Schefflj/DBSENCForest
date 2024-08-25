import numpy as np
import pandas as pd

from DBSENCForest.data_structure.ext_dataframe import ExtDataFrame
from DBSENCForest.data_structure.senc_forest.senc_forest import SENCForest
from DBSENCForest.data_structure.senc_forest.senc_forests_algorithm import SencForestsAlgorithm
from DBSENCForest.multiple_ncd.multiple_ncd import Multiple_NCD


class DBSENCForest():

    def __init__(self, training_data, forest_size, sub_set_size, buffer_size, split_size, sim_matrix_given, minimize_error, mncd_test):
        """
        Initializes the DBSENCForest class with the specified parameters.

        :param training_data: pandas DataFrame, The initial training data.
        :param forest_size: int, The size of the forest (number of trees).
        :param sub_set_size: int, The size of the subset used for each tree.
        :param buffer_size: int, The size of the buffer before retraining.
        :param split_size: int, The split size used during training.
        :param sim_matrix_given: bool, Whether a similarity matrix is provided.
        :param minimize_error: str, The strategy to minimize error ('FP', 'FN').
        :param mncd_test: bool, Whether to use the MNCD test for novel class detection.
        """
        self.mncd_test = mncd_test
        self.ext_training_data = ExtDataFrame(training_data)
        self.forest_size = forest_size
        self.sub_set_size = sub_set_size
        self.buffer_size = buffer_size
        self.split_size = split_size


        num_dim = self.ext_training_data.get_feature_samples_numpy().shape[1]
        rseed = int(np.sum(100 * np.random.rand()))

        self.senc_algorithm = SencForestsAlgorithm()

        self.forest = SENCForest(self.ext_training_data.get_feature_samples_numpy(), forest_size, sub_set_size, num_dim, rseed, self.ext_training_data.get_encoded_labels_numpy())

        self.forest_without_pseudo = self.forest

        self.retrain_buffer = []

        self.new_class_buffer = []

        self.new_class_buffer_df = ExtDataFrame(training_data.copy().iloc[0:0])

        self.unite_buffer = {}

        self.origin_class_set = list(self.ext_training_data.label_mapping.values())

        self.similarity_matrix = None
        self.all_nc_samples = None

        self.initial_training_df = self.ext_training_data.dataframe.copy()
        self.initial_forest = SENCForest(self.ext_training_data.get_feature_samples_numpy(), forest_size, sub_set_size, num_dim, rseed, self.ext_training_data.get_encoded_labels_numpy())

        self.rest = False
        self.already_novel_classes = False

        self.sim_matrix_given = sim_matrix_given

        self.minimize_error = minimize_error

        self.new_prediction = False

    def send_stream(self, data_stream):
        """
        Processes the data stream, making predictions and updating the model as needed.

        :param data_stream: pandas DataFrame, The incoming data stream to be processed.
        :return: pandas DataFrame, The updated training data with new predictions.
        """
        ext_data_stream = ExtDataFrame(data_stream)

        self.counter = 0
        for entry in ext_data_stream.sample_dict.values():

            prediction = self.predict(entry.row)
            pred_label = prediction[0]
            if pred_label in self.ext_training_data.label_mapping.values() and not self.mncd_test:
                entry.notes += f"1. known label predicted: {self.ext_training_data.invert_label_mapping[pred_label]}\n"
                self.retrain_buffer.append((entry, prediction[2][:, 3], self.ext_training_data.invert_label_mapping[pred_label]))
                print(entry.notes)

            if pred_label == -2 or self.mncd_test:
                self.counter += 1
                entry.notes += f"1. new class detected\n"
                self.new_class_buffer.append((entry, prediction[2][:, 3], -2))
                print(entry.notes)

            if self.counter > 20:
                self.new_prediction = True
            self.update_model()


        self.rest = True
        self.update_model()
        return self.ext_training_data.dataframe



    def predict(self, entry):
        """
        Predicts the label for a single data point.

        :param entry: pandas Series, A single data entry for prediction.
        :return: tuple, The predicted label, scores, and mass from the SENC algorithm.
        """
        new_class_label = -2

        para = {
            'alpha': 1,
            'buffersize': len(self.new_class_buffer)
        }


        entry = entry.apply(pd.to_numeric, errors='coerce').to_numpy()

        mass, mtimetest = self.senc_algorithm.sence_estimation(entry.reshape(1, -1), self.forest, para['alpha'], self.forest.anomaly)
        answermass = mass[:, 2].copy()
        answermass[mass[:, 4] == 1] = new_class_label
        scores = self.senc_algorithm.tabulate(answermass)
        score_1 = scores[scores[:, 1] == np.max(scores[:, 1]), :]
        return (int(score_1[0][0]), scores, mass)




    def update_model(self):
        """
        Updates the model based on the current state of the buffers, retraining the forest if necessary.
        """

        change = False

        if len(self.retrain_buffer) >= self.buffer_size and self.new_prediction == True:
            change = True

            for entry in self.retrain_buffer:
                entry[0].row.loc[self.ext_training_data.dataframe.columns[-1]] = entry[2]
                initial_labels = set(self.initial_training_df[self.initial_training_df.columns[-1]])
                if entry[2] in initial_labels:
                    self.initial_training_df = pd.concat([self.initial_training_df, pd.DataFrame([entry[0].row])])
                else:
                    self.ext_training_data.add_data(pd.DataFrame([entry[0].row]))


            self.retrain_buffer = []
            self.new_prediction = False

        if (len(self.new_class_buffer) >= self.buffer_size or (self.rest == True and len(self.new_class_buffer) + self.ext_training_data.dataframe.iloc[:, -1].str.count("pseudo").sum() >= self.buffer_size)) and self.new_prediction == True:
            print("run dmnc")
            self.already_novel_classes = True
            change = True


            for entry in self.new_class_buffer:

                entry[0].row.loc[self.ext_training_data.dataframe.columns[-1]] = -2

                self.new_class_buffer_df.add_data(pd.DataFrame([entry[0].row]))


            mncd = Multiple_NCD(self.sim_matrix_given, self.ext_training_data, self.new_class_buffer_df, self.initial_training_df, self, self.minimize_error)
            mncd_df = mncd.multiple_novel_class_detection()

            self.ext_training_data = ExtDataFrame(pd.concat([self.initial_training_df, mncd_df]))

            self.counter = 0
            self.new_prediction = False

        if not self.new_prediction and self.rest:

            for entry in self.new_class_buffer:
                entry[0].row.loc[self.ext_training_data.dataframe.columns[-1]] = -2

                self.ext_training_data.add_data(pd.DataFrame([entry[0].row]))

            for entry in self.retrain_buffer:
                try:
                    entry[0].row.loc[self.ext_training_data.dataframe.columns[-1]] = self.ext_training_data.invert_label_mapping[self.predict(entry[0].row)[0]]
                except:
                    entry[0].row.loc[self.ext_training_data.dataframe.columns[-1]] = self.predict(entry[0].row)[0]

                self.ext_training_data.add_data(pd.DataFrame([entry[0].row]))


        if change and not self.rest:
            self.forest = self.recreate_forest(self.ext_training_data.dataframe.copy(), {})



    def recreate_forest(self, df, ignore_labels):
        """
        Recreates the forest, optionally ignoring specified labels during training.

        :param df: pandas DataFrame, The data used to recreate the forest.
        :param ignore_labels: dict, Labels to ignore during the forest recreation.
        :return: SENCForest, The newly created SENCForest instance.
        """

        for label in ignore_labels:
            df = df[df[df.columns[-1]] != label]

        ext_temp_training_data = ExtDataFrame(df)

        num_dim = self.ext_training_data.get_feature_samples_numpy().shape[1]
        rseed = int(np.sum(100 * np.random.rand()))
        return SENCForest(ext_temp_training_data.get_feature_samples_numpy(), self.forest_size,
                                 self.sub_set_size,
                                 num_dim, rseed, ext_temp_training_data.get_encoded_labels_numpy())

    def match_labels(self, df1, df2):
        """
        Matches labels between two DataFrames.

        :param df1: pandas DataFrame, The first DataFrame with features and labels.
        :param df2: pandas DataFrame, The second DataFrame with features and labels.
        :return: tuple, Two lists containing the matched labels from each DataFrame.
        """
        n_minus_1 = df1.shape[1] - 1

        df1_sorted = df1.sort_values(by=df1.columns[:n_minus_1].tolist()).reset_index(drop=True)
        df2_sorted = df2.sort_values(by=df2.columns[:n_minus_1].tolist()).reset_index(drop=True)

        labels_df1 = []
        labels_df2 = []

        for i, row1 in df1_sorted.iterrows():
            match = df2_sorted[(df2_sorted.iloc[:, :-1] == row1.iloc[:-1].values).all(axis=1)]
            if not match.empty:
                labels_df1.append(row1.iloc[-1])
                labels_df2.append(match.iloc[0, -1])

        return labels_df1, labels_df2



    def create_node_id_lists(self, ext_all_nc_df):
        """
        Creates lists of node IDs for each entry in the data.

        :param ext_all_nc_df: ExtDataFrame, An extended DataFrame containing all novel class data.
        :return: list, A list of node ID lists for each data point.
        """

        node_id_lists = []

        for entry in ext_all_nc_df.sample_dict.values():

            entry = entry.row.iloc[:-1]

            entry = entry.apply(pd.to_numeric, errors='coerce').to_numpy()

            self.senc_algorithm = SencForestsAlgorithm()

            mass, mtimetest = self.senc_algorithm.sence_estimation(entry.reshape(1, -1), self.forest_without_pseudo, 1,
                                                                   self.forest_without_pseudo.anomaly)
            node_ids = mass[:, 3].copy()

            node_id_lists.append(node_ids)

        return node_id_lists



