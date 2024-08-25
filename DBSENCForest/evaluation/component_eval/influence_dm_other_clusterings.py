import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

from DBSENCForest.data_structure.ext_dataframe import ExtDataFrame
from DBSENCForest.evaluation.art_datasets.art_dataset1_generator import Dataset1Generator
from DBSENCForest.evaluation.art_datasets.art_dataset2_generator import Dataset2Generator
from DBSENCForest.evaluation.art_datasets.art_dataset3_generator import Dataset3Generator
from DBSENCForest.evaluation.art_datasets.art_dataset4_generator import Dataset4Generator
from DBSENCForest.evaluation.data_manipulation import DataManipulation
from DBSENCForest.main.data_stream_classifier import DBSENCForest
from DBSENCForest.multiple_ncd.agglo import HierarchicalClusteringMethods
from DBSENCForest.multiple_ncd.dissimmatrix import Dissimmatrix
from DBSENCForest.multiple_ncd.kmedoids import KMedoids_methods


class InfluenceDmOtherClusterings():

    def plot_ari_barchart(self, ari_values, dataset_name):
        """
        Plots a bar chart comparing Adjusted Rand Index (ARI) values for different clustering methods,
        with and without the use of a similarity matrix.

        :param ari_values: list, ARI values to plot (expected order: [KMedoids with matrix, KMedoids without matrix,
                          AHC with matrix, AHC without matrix, DBSCAN with matrix, DBSCAN without matrix]).
        :param dataset_name: str, the name of the dataset for the title of the plot.
        """
        categories = ['KMedoids', 'AHC', 'DBSCAN']
        methods = ['With DistMatrix', 'Without DistMatrix']

        ari_values = np.array(ari_values).reshape(3, 2)

        x = np.arange(len(categories))
        width = 0.25

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width / 2, ari_values[:, 0], width, label=methods[0], color='b')
        bars2 = ax.bar(x + width / 2, ari_values[:, 1], width, label=methods[1], color='r')

        ax.set_ylim(0, 1.5)

        ax.set_xlabel('Clustering Method')
        ax.set_ylabel('Adjusted Rand Index (ARI)')
        ax.set_title(f'ARI Comparison on {dataset_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)

        fig.tight_layout()

        plt.show()

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


    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates evaluation metrics including Jaccard Index, F1 Score, False Alarm Rate, and Adjusted Rand Index
        between the ground truth labels and the predicted labels.

        :param y_true: list or numpy array, ground truth labels.
        :param y_pred: list or numpy array, predicted labels.
        :return: tuple, containing Jaccard Index, F1 Score, False Alarm Rate, and Adjusted Rand Index.
        """
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

    def compare_kmedoids_versions(self, training_data, novel_class_data, n_clusters):
        """
        Compares the performance of KMedoids clustering with and without a precomputed similarity matrix.

        :param training_data: pandas DataFrame, training data.
        :param novel_class_data: pandas DataFrame, novel class data for clustering.
        :param n_clusters: int, number of clusters.
        :return: tuple, containing metrics (Jaccard, F1, False Alarm Rate, Adjusted Rand Index) for both versions.
        """
        ext_novel_class_data = ExtDataFrame(novel_class_data)
        ground_truth_list = novel_class_data.iloc[:, -1].tolist()
        dissimmatrix = Dissimmatrix()


        novel_class_data_without_label = novel_class_data.copy()
        novel_class_data_without_label.pop(novel_class_data.columns[-1])

        cluster_list_normal = KMedoids_methods.kmedoids_clustering(novel_class_data_without_label, n_clusters)
        self.plot_ground_truth_vs_predictions(novel_class_data_without_label, ground_truth_list,
                                              cluster_list_normal)

        scores_normal = self.calculate_metrics(ground_truth_list, cluster_list_normal)

        tr_df = training_data.copy()
        tr_df[tr_df.columns[-1]] = 1
        testsenc = DBSENCForest(tr_df, 100, 120, 20, 10, True, "", True)
        node_id_lists = testsenc.create_node_id_lists(ext_novel_class_data)
        sim_matrix = dissimmatrix.calculate_similarity_matrix(node_id_lists)

        cluster_list_sim_matrix = KMedoids_methods.kmedoids_clustering_with_sim_matrix(sim_matrix, n_clusters)

        scores_matrix = self.calculate_metrics(ground_truth_list, cluster_list_sim_matrix)

        print(scores_normal)

        print(scores_matrix)

        return scores_normal, scores_matrix

    def compare_agglo_versions(self, training_data, novel_class_data, n_clusters):
        """
        Compares the performance of Agglomerative Hierarchical Clustering (AHC) with and without a precomputed similarity matrix.

        :param training_data: pandas DataFrame, training data.
        :param novel_class_data: pandas DataFrame, novel class data for clustering.
        :param n_clusters: int, number of clusters.
        :return: tuple, containing metrics (Jaccard, F1, False Alarm Rate, Adjusted Rand Index) for both versions.
        """
        ext_novel_class_data = ExtDataFrame(novel_class_data)
        ground_truth_list = novel_class_data.iloc[:, -1].tolist()
        dissimmatrix = Dissimmatrix()


        novel_class_data_without_label = novel_class_data.copy()
        novel_class_data_without_label.pop(novel_class_data.columns[-1])

        cluster_list_normal = HierarchicalClusteringMethods.hierarchical_clustering(novel_class_data_without_label,
                                                                                    n_clusters)
        self.plot_ground_truth_vs_predictions(novel_class_data_without_label, ground_truth_list,
                                              cluster_list_normal)

        scores_normal = self.calculate_metrics(ground_truth_list, cluster_list_normal)

        tr_df = training_data.copy()
        tr_df[tr_df.columns[-1]] = 1
        testsenc = DBSENCForest(tr_df, 100, 120, 20, 10, True, "", True)
        node_id_lists = testsenc.create_node_id_lists(ext_novel_class_data)
        sim_matrix = dissimmatrix.calculate_similarity_matrix(node_id_lists)

        cluster_list_sim_matrix = HierarchicalClusteringMethods.hierarchical_clustering_with_sim_matrix(sim_matrix,
                                                                                                        n_clusters)

        scores_matrix = self.calculate_metrics(ground_truth_list, cluster_list_sim_matrix)

        print(scores_normal)

        print(scores_matrix)

        return scores_normal, scores_matrix



if __name__ == "__main__":

    dataset_list = []


    datagen = Dataset1Generator()
    dataset_list.append(datagen.get_dataframe())

    datagen = Dataset2Generator()
    dataset_list.append(datagen.get_dataframe())

    datagen = Dataset3Generator()
    dataset_list.append(datagen.get_dataframe())

    datagen = Dataset4Generator()
    dataset_list.append(datagen.get_dataframe())

    testa = DataManipulation()

    ari_dbscan_normal = [0.4416, 0.8185, 0.8535, 0.9917]
    ari_dbscan_matrix = [1.0, 0.9605, 0.9958, 0.9958]
    datasets = ["Art_Df1", "Art_Df2", "Art_Df3", "Art_Df4"]
    i = 0

    for data in dataset_list:
        data = data.dropna()
        data, scaler = testa.normalize_features(data)

        tests = InfluenceDmOtherClusterings()


        medoids_metrics = tests.compare_kmedoids_versions(data.copy(), data.copy(), len(set(data.iloc[:, -1].tolist())))
        agglo_metrics = tests.compare_agglo_versions(data.copy(), data.copy(), len(set(data.iloc[:, -1].tolist())))

        ari_values = [medoids_metrics[1][3], medoids_metrics[0][3], agglo_metrics[1][3], agglo_metrics[0][3], ari_dbscan_matrix[i], ari_dbscan_normal[i]]
        tests.plot_ari_barchart(ari_values, datasets[i])

        i += 1


