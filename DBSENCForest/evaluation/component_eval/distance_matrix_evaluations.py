import random
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter

from DBSENCForest.data_structure.ext_dataframe import ExtDataFrame
from DBSENCForest.evaluation.art_datasets.art_dataset1_generator import Dataset1Generator
from DBSENCForest.evaluation.art_datasets.art_dataset2_generator import Dataset2Generator
from DBSENCForest.evaluation.art_datasets.art_dataset3_generator import Dataset3Generator
from DBSENCForest.evaluation.art_datasets.art_dataset4_generator import Dataset4Generator
from DBSENCForest.evaluation.data_manipulation import DataManipulation
from DBSENCForest.main.data_stream_classifier import DBSENCForest
from DBSENCForest.multiple_ncd.dbscan import DBSCAN_methods
from DBSENCForest.multiple_ncd.dissimmatrix import Dissimmatrix


class DistanceMatrixEvaluations:

    def plot_ground_truth_vs_predictions(self, df, ground_truth_labels, predicted_labels):
        """
        Plots a comparison between ground truth labels and predicted labels using a 2D scatter plot.
        If the input DataFrame has more than two dimensions, it uses t-SNE to reduce dimensions to 2.

        :param df: pandas DataFrame, the input dataset.
        :param ground_truth_labels: list or numpy array, ground truth labels for the data.
        :param predicted_labels: list or numpy array, predicted labels for the data.
        """
        if df.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            df_2d = tsne.fit_transform(df)
        else:
            df_2d = df.values

        df_2d = pd.DataFrame(df_2d, columns=['Dim1', 'Dim2'])

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Dim1', y='Dim2', hue=ground_truth_labels, data=df_2d, palette="viridis")
        plt.title('Ground Truth Labels')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x='Dim1', y='Dim2', hue=predicted_labels, data=df_2d, palette="viridis")
        plt.title('Predicted Labels')

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, df, predicted_labels_sim, predicted_labels_normal, threshold=20):
        """
        Plots the comparison between predicted labels with and without the use of a similarity matrix.
        It also maps rare labels (appearing less than `threshold` times) to -1.

        :param df: pandas DataFrame, the input dataset.
        :param predicted_labels_sim: list or numpy array, predicted labels with similarity matrix.
        :param predicted_labels_normal: list or numpy array, predicted labels without similarity matrix.
        :param threshold: int, threshold for determining rare labels.
        """
        def map_rare_labels(labels, threshold):
            label_counts = Counter(labels)
            return [label if label_counts[label] > threshold else -1 for label in labels]

        predicted_labels_sim_mapped = map_rare_labels(predicted_labels_sim, threshold)
        predicted_labels_normal_mapped = map_rare_labels(predicted_labels_normal, threshold)

        if df.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            df_2d = tsne.fit_transform(df)
        else:
            df_2d = df.values

        df_2d = pd.DataFrame(df_2d, columns=['Dim1', 'Dim2'])

        # Plot 1: Predicted Labels without distance matrix
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Dim1', y='Dim2', hue=predicted_labels_normal_mapped, data=df_2d, palette="viridis")
        plt.title('Predicted Labels without distance matrix')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x='Dim1', y='Dim2', hue=predicted_labels_sim_mapped, data=df_2d, palette="viridis")
        plt.title('Predicted Labels with distance matrix')

        plt.tight_layout()
        plt.show()

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates evaluation metrics including Jaccard Index, F1 Score, False Alarm Rate, and Adjusted Rand Index
        between the ground truth labels and the predicted labels.

        :param y_true: list or numpy array, ground truth labels.
        :param y_pred: list or numpy array, predicted labels.
        :return: tuple, containing Jaccard Index, F1 Score, False Alarm Rate, and Adjusted Rand Index.
        """
        y_pred = [random.randint(10 ** 9, 10 ** 12) if x == -1 else x for x in y_pred]

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        unique_true_labels = np.unique(y_true)
        unique_pred_labels = np.unique(y_pred)

        true_label_mapping = {label: idx for idx, label in enumerate(unique_true_labels)}
        pred_label_mapping = {label: idx for idx, label in enumerate(unique_pred_labels)}

        y_true_mapped = np.array([true_label_mapping[label] for label in y_true])
        y_pred_mapped = np.array([pred_label_mapping[label] for label in y_pred])

        cm = confusion_matrix(y_true_mapped, y_pred_mapped)

        row_ind, col_ind = linear_sum_assignment(-cm)

        optimal_y_pred_mapped = np.zeros_like(y_pred_mapped)
        for i, j in zip(row_ind, col_ind):
            optimal_y_pred_mapped[y_pred_mapped == j] = i

        jaccard = jaccard_score(y_true_mapped, optimal_y_pred_mapped, average='macro')

        f1 = f1_score(y_true_mapped, optimal_y_pred_mapped, average='macro')

        cm = confusion_matrix(y_true_mapped, optimal_y_pred_mapped)
        fp = cm.sum(axis=0) - np.diag(cm)
        tn = cm.sum() - (cm.sum(axis=1) + fp + np.diag(cm))
        false_alarm_rate = fp.sum() / (fp.sum() + tn.sum())

        adjusted_rand = adjusted_rand_score(y_true_mapped, optimal_y_pred_mapped)

        return jaccard, f1, false_alarm_rate, adjusted_rand

    def grid_search_dbscan_params(self, data, ground_truth, eps_range, min_samples_range):
        """
        Performs grid search to find the best parameters for DBSCAN clustering on the raw data.

        :param data: numpy array or pandas DataFrame, input data for clustering.
        :param ground_truth: list or numpy array, ground truth labels.
        :param eps_range: iterable, range of epsilon values to search.
        :param min_samples_range: iterable, range of min_samples values to search.
        :return: tuple, containing the best min_samples and epsilon values.
        """
        best_score = -1
        best_params = None

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)
                labels = [random.randint(10 ** 9, 10 ** 12) if x == -1 else x for x in labels]

                if len(set(labels)) > 1 and len(set(labels)) < len(data):
                    score = adjusted_rand_score(ground_truth, labels)
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)

        return best_params[1], best_params[0]

    def grid_search_dbscan_params_precomputed(self, sim_matrix, ground_truth, eps_range, min_samples_range):
        """
        Performs grid search to find the best parameters for DBSCAN clustering with a precomputed similarity matrix.

        :param sim_matrix: numpy array, precomputed similarity matrix.
        :param ground_truth: list or numpy array, ground truth labels.
        :param eps_range: iterable, range of epsilon values to search.
        :param min_samples_range: iterable, range of min_samples values to search.
        :return: tuple, containing the best min_samples and epsilon values.
        """
        best_score = -1
        best_params = None

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                labels = dbscan.fit_predict(sim_matrix)
                labels = [random.randint(10 ** 9, 10 ** 12) if x == -1 else x for x in labels]

                if len(set(labels)) > 1 and len(set(labels)) < len(sim_matrix):
                    score = adjusted_rand_score(ground_truth, labels)
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)

        return best_params[1], best_params[0]

    def compare_dbscan_versions(self, training_data, novel_class_data):
        """
        Compares the performance of DBSCAN clustering with and without a precomputed similarity matrix.

        :param training_data: pandas DataFrame, training data.
        :param novel_class_data: pandas DataFrame, novel class data for clustering.
        :return: tuple, containing scores (Jaccard, F1, False Alarm Rate, Adjusted Rand Index) for each DBSCAN version.
        """
        ext_novel_class_data = ExtDataFrame(novel_class_data)
        ground_truth_list = novel_class_data.iloc[:, -1].tolist()
        dissimmatrix = Dissimmatrix()

        novel_class_data_without_label = novel_class_data.copy()
        novel_class_data_without_label.pop(novel_class_data.columns[-1])

        min_pts, epsilon = self.grid_search_dbscan_params(novel_class_data_without_label, ground_truth_list,
                                                          np.arange(0.01, 10, 0.01), range(2, 20))
        print(f"Optimized min_pts: {min_pts}, epsilon: {epsilon}")

        cluster_list_normal = DBSCAN_methods.dbscan_clustering(novel_class_data_without_label, epsilon, min_pts)

        scores_normal = self.calculate_metrics(ground_truth_list, cluster_list_normal.copy())

        tr_df = training_data.copy()
        tr_df[tr_df.columns[-1]] = 1
        testsenc = DBSENCForest(tr_df, 100, 120, 20, 10, True, "", True)
        node_id_lists = testsenc.create_node_id_lists(ext_novel_class_data)

        sim_matrix = dissimmatrix.calculate_similarity_matrix(node_id_lists)

        min_pts, epsilon = self.grid_search_dbscan_params_precomputed(sim_matrix, ground_truth_list,
                                                                      np.arange(0.01, 10, 0.01), range(2, 20))
        print(f"Optimized min_pts: {min_pts}, epsilon: {epsilon}")

        cluster_list_sim_matrix = DBSCAN_methods.dbscan_clustering_with_sim_matrix(sim_matrix, epsilon, min_pts)
        scores_matrix = self.calculate_metrics(ground_truth_list, cluster_list_sim_matrix.copy())

        self.plot_predictions(novel_class_data_without_label, cluster_list_sim_matrix, cluster_list_normal)

        print(scores_normal)
        print(scores_matrix)

        return scores_normal, scores_matrix



if __name__ == "__main__":

    dataset_list = []

    datagen = Dataset4Generator()
    dataset_list.append(datagen.get_dataframe())

    datagen = Dataset1Generator()
    dataset_list.append(datagen.get_dataframe())

    datagen = Dataset2Generator()
    dataset_list.append(datagen.get_dataframe())

    datagen = Dataset3Generator()
    dataset_list.append(datagen.get_dataframe())


    testa = DataManipulation()



    for data in dataset_list:

        data = data.dropna()
        data, scaler = testa.normalize_features(data)

        tests = DistanceMatrixEvaluations()

        dbscan_metrics = tests.compare_dbscan_versions(data.copy(), data.copy())
