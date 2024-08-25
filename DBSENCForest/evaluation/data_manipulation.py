import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class DataManipulation:

    def gaussian_normalize_except_last(self, df):
        """
        Performs Gaussian normalization (z-score normalization) on all columns of the DataFrame except the last one.

        :param df: pandas DataFrame, The input DataFrame with features and a label column.
        :return: pandas DataFrame, The DataFrame with normalized features.
        """
        df_normalized = df.copy()

        columns_to_normalize = df.columns[:-1]

        for column in columns_to_normalize:
            mean = df[column].mean()
            std = df[column].std()
            df_normalized[column] = (df[column] - mean) / std

        return df_normalized

    def normalize_features(self, data):
        """
        Applies Min-Max normalization to the feature columns of the DataFrame except the last column.

        :param data: pandas DataFrame, The input DataFrame with features and a label column.
        :return: tuple, (normalized DataFrame, fitted MinMaxScaler object)
        """
        normalized_data = data.copy()

        features = normalized_data.iloc[:, :-1]

        scaler = MinMaxScaler()

        normalized_features = scaler.fit_transform(features)

        normalized_data.iloc[:, :-1] = normalized_features

        return normalized_data, scaler

    def plot_pca_2d(self, df):
        """
        Performs PCA to reduce the feature space to 2 dimensions and plots the result.

        :param df: pandas DataFrame, The input DataFrame with features and a label column.
        :return: None
        """

        labels = df.iloc[:, -1]
        data = df.iloc[:, :-1]


        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)

        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['Label'] = labels.values

        plt.figure(figsize=(10, 7))
        unique_labels = pca_df['Label'].unique()
        for label in unique_labels:
            indices_to_plot = pca_df['Label'] == label
            plt.scatter(pca_df.loc[indices_to_plot, 'PC1'],
                        pca_df.loc[indices_to_plot, 'PC2'],
                        label=label)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA Plot')
        plt.legend()
        plt.grid()
        plt.show()

    def delete_entries_except_first_n(self, df, label, n):
        """
        Deletes all but the first 'n' entries of the specified label from the DataFrame.

        :param df: pandas DataFrame, The input DataFrame with features and a label column.
        :param label: str, The label for which all but the first 'n' entries will be deleted.
        :param n: int, The number of entries to keep for the specified label.
        :return: pandas DataFrame, The DataFrame after deletion.
        """
        filtered_df = df[df.iloc[:, -1] == label]

        kept_entries = filtered_df.head(n)

        df_filtered = df[df.iloc[:, -1] != label]

        df_final = pd.concat([df_filtered, kept_entries])

        df_final.reset_index(drop=True, inplace=True)

        return df_final


    def delete_entries(self, df, label_to_delete, n_to_delete):
        """
        Deletes 'n_to_delete' random entries of the specified label from the DataFrame.

        :param df: pandas DataFrame, The input DataFrame with features and a label column.
        :param label_to_delete: str, The label for which 'n_to_delete' entries will be deleted.
        :param n_to_delete: int, The number of entries to delete for the specified label.
        :return: tuple, (modified DataFrame, DataFrame with deleted entries)
        """

        deleted_entries = df[df.iloc[:, -1] == label_to_delete]
        modified_df = df[df.iloc[:, -1] != label_to_delete]
        deleted_entries.reset_index(drop=True, inplace=True)
        modified_df.reset_index(drop=True, inplace=True)



        splitter = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=None, random_state=42)

        X = modified_df.iloc[:, :-1]
        y = modified_df.iloc[:, -1]

        try:
            for train_index, _ in splitter.split(X, y):
                sampled_indices = random.sample(list(train_index), n_to_delete)
                deleted_random_entries = modified_df.loc[sampled_indices]
                modified_df = modified_df.drop(sampled_indices)

            del_entries = pd.concat([deleted_entries, deleted_random_entries])
            del_entries.reset_index(drop=True, inplace=True)
            modified_df.reset_index(drop=True, inplace=True)

            return modified_df, del_entries
        except:
            return modified_df, deleted_entries


    def get_polluted_df(self, label, data):
        """
        Replaces the specified label in the DataFrame with 'St 48' in the last column.

        :param label: str, The label to be replaced in the DataFrame.
        :param data: pandas DataFrame, The input DataFrame with features and a label column.
        :return: pandas DataFrame, The DataFrame with the label replaced.
        """

        dataframe = data

        last_column_index = dataframe.shape[1] - 1

        zu_ersetzendes_label = label

        unique_labels_without_target = [label for label in dataframe.iloc[:, last_column_index].unique() if label != zu_ersetzendes_label]


        indices_to_replace = dataframe.index[dataframe.iloc[:, last_column_index] == zu_ersetzendes_label].tolist()

        for index in indices_to_replace:
            dataframe.at[index, 'steel_class'] = 'St 48'

        dataframe.replace(',', '.', regex=True, inplace=True)

        return dataframe

    def plot_data(self, data):
        """
        Plots the t-SNE visualization of the data, encoding labels for color coding.

        :param data: pandas DataFrame, The input DataFrame with features and a label column.
        :return: None
        """
        data = data.copy()
        encoder = LabelEncoder()
        last_column = data.columns[-1]
        data['Encoded_Label'] = encoder.fit_transform(
            data[last_column])
        data = data.drop([last_column], axis=1)
        numerische_werte = data.iloc[:, :-1]
        labels = data.iloc[:, -1]
        print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
        tsne = TSNE(n_components=2, random_state=42)
        reduzierte_werte = tsne.fit_transform(numerische_werte)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduzierte_werte[:, 0], reduzierte_werte[:, 1], c=labels, cmap='viridis')
        plt.title('t-SNE Plot mit Labels')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar(label='Label')
        plt.show()