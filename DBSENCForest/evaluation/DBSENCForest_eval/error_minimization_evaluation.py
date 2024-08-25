import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


import itertools

import pandas as pd

from sklearn.metrics import adjusted_rand_score, confusion_matrix
import warnings
import seaborn as sns

from DBSENCForest.evaluation.art_datasets.art_dataset1_generator import Dataset1Generator
from DBSENCForest.evaluation.art_datasets.art_dataset2_generator import Dataset2Generator
from DBSENCForest.evaluation.art_datasets.art_dataset3_generator import Dataset3Generator
from DBSENCForest.evaluation.art_datasets.art_dataset4_generator import Dataset4Generator
from DBSENCForest.evaluation.data_manipulation import DataManipulation
from DBSENCForest.main.data_stream_classifier import DBSENCForest

warnings.filterwarnings("ignore")
class ErrorMinimizationEvaluation():

    def plot_ground_truth_vs_predictions(self, df, training_df, ground_truth_labels, predicted_labels):
        """
        Plots the ground truth labels versus predicted labels using t-SNE for dimensionality reduction.

        :param df: pandas DataFrame, New data to be plotted.
        :param training_df: pandas DataFrame, Training data used for predictions.
        :param ground_truth_labels: list, Ground truth labels for the new data.
        :param predicted_labels: list, Predicted labels for the new data.
        """

        pre_labels = training_df.iloc[:, -1].to_list()

        ground_truth_labels = pre_labels + ground_truth_labels
        predicted_labels = pre_labels + predicted_labels

        df_combined = pd.concat([training_df, df])
        df_combined.pop(df_combined.columns[-1])

        df = pd.concat([training_df, df])
        df.pop(df.columns[-1])

        if df.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            df_2d = tsne.fit_transform(df)
        else:
            df_2d = df.values

        df_2d = pd.DataFrame(df_2d, columns=['Dim1', 'Dim2'])

        plt.figure(figsize=(10, 7))

        unique_labels = len(set(predicted_labels))
        palette = sns.color_palette("tab10", unique_labels)

        plt.subplot(1, 1, 1)
        sns.scatterplot(x='Dim1', y='Dim2', hue=predicted_labels, data=df_2d, palette=palette)
        plt.title('Predicted Labels')

        plt.tight_layout()
        plt.show()

    def calculate_precision_recall(self, ground_truth, predictions):
        """
        Calculates precision and recall based on the confusion matrix for each class.

        :param ground_truth: list or array, True labels.
        :param predictions: list or array, Predicted labels.
        :return: tuple, (average precision, average recall) values.
        """
        conf_matrix = confusion_matrix(ground_truth, predictions)

        precisions = []
        recalls = []

        for j in range(conf_matrix.shape[1]):
            tp = np.max(conf_matrix[:, j])
            matched_class_index = np.argmax(conf_matrix[:, j])

            fp = np.sum(conf_matrix[:, j]) - tp
            fn = np.sum(
                conf_matrix[matched_class_index, :]) - tp

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        return avg_precision, avg_recall

    def evaluate(self, predicted, ground_truth):
        """
        Evaluates the performance of the predictions against the ground truth using precision, recall, F1-score,
        and Adjusted Rand Index.

        :param predicted: list or array, Predicted labels.
        :param ground_truth: list or array, Ground truth labels.
        :return: tuple, (precision, recall, F1-score, adjusted_rand_score) values.
        """

        l1 = predicted

        l2 = ground_truth


        l1 = [random.randint(10 ** 9, 10 ** 12) if x == -2 else x for x in l1]

        le = LabelEncoder()
        y_true = le.fit_transform(l2)
        y_pred = le.fit_transform(l1)

        conf_matrix = confusion_matrix(y_true, y_pred)

        row_ind, col_ind = linear_sum_assignment(-conf_matrix)

        y_pred_mapped = np.zeros_like(y_pred)
        for i, j in zip(row_ind, col_ind):
            y_pred_mapped[y_pred == j] = i



        precision1, recall1 = self.calculate_precision_recall(y_true, y_pred_mapped)

        f1 = 2 * ((precision1 * recall1) / (precision1 + recall1))

        adjusted_rand = adjusted_rand_score(y_true, y_pred_mapped)


        return (precision1, recall1, f1, adjusted_rand)

    def run_DBSENCForest(self, train_data, new_data, error_min):
        """
        Runs the DBSENCForest algorithm on the given training and new data, then evaluates the results.

        :param train_data: pandas DataFrame, The training data used to build the initial DBSENCForest model.
        :param new_data: pandas DataFrame, The new data to be predicted and evaluated.
        :param error_min: str, Specifies the error minimization strategy ('FP' for minimizing false positives or 'FN' for minimizing false negatives).

        :return: tuple, (precision, recall, F1-score, adjusted_rand_score) values.
                 If no new classes are recognized, returns (1, 0, 0, 0).
                 If there are no filtered labels, returns (1, 1, 1, 1).
        """
        origin_new_data = new_data.copy()
        new_data.pop(new_data.columns[-1])

        origin_train_data = train_data.copy()

        testsenc = DBSENCForest(train_data, 100, 50, len(new_data), 5, True, error_min, True)

        predicted_df = testsenc.send_stream(new_data).tail(len(origin_new_data))
        predicted_df.iloc[:, :-1] = predicted_df.iloc[:, :-1].apply(pd.to_numeric)
        predicted_df = predicted_df.sort_values(by=list(predicted_df.columns[:-1]))

        ground_truth_df = origin_new_data
        ground_truth_df.iloc[:, :-1] = ground_truth_df.iloc[:, :-1].apply(pd.to_numeric)
        ground_truth_df = ground_truth_df.sort_values(by=list(ground_truth_df.columns[:-1]))

        l1, l2 = testsenc.match_labels(predicted_df, ground_truth_df)

        filtered_l1 = []
        filtered_l2 = []
        pseudo_counter = 0

        for item1, item2 in zip(l1, l2):
            if (isinstance(item1, str) and "pseudo" in item1) or (isinstance(item1, float) and item1 < 0):
                filtered_l1.append(item1)
                filtered_l2.append(item2)
            if (isinstance(item1, str) and "pseudo" in item1):
                pseudo_counter += 1



        if filtered_l1 == []:
            return (1, 1, 1, 1)
        if pseudo_counter == 0:
            print("keine neuen Klassen erkannt")
            return (1, 0, 0, 0)


        print(len(filtered_l1) / len(l1) * 100, "% korrekt als Neuheiten erkannt")
        print(pseudo_counter / len(filtered_l1) * 100, "% der Neuheiten als Teil neuer Klassen erkannt")


        eval_sim = self.evaluate(filtered_l1, filtered_l2)

        return eval_sim

    def filter_rare_labels(self, df, threshold):
        """
        Filters out labels in a DataFrame that occur less frequently than the specified threshold.

        :param df: pandas DataFrame, DataFrame containing labels to filter.
        :param threshold: int, The frequency threshold below which labels are removed.
        :return: pandas DataFrame, Filtered DataFrame.
        """
        last_column = df.columns[-1]

        label_counts = df[last_column].value_counts()

        rare_labels = label_counts[label_counts < threshold].index

        filtered_df = df[~df[last_column].isin(rare_labels)]

        return filtered_df


def perform_tsne_and_plot(data, name):
    """
    Performs t-SNE dimensionality reduction on the given data and plots the results.

    :param data: pandas DataFrame, The dataset to be visualized.
    :param name: str, The name of the dataset for the plot title.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data.iloc[:, :-1])

    labels = data.iloc[:, -1].unique()

    colors = plt.cm.get_cmap('viridis', len(labels))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        indices = data.iloc[:, -1] == label
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                    color=colors(i), label=str(label), edgecolor='k', s=50)

    plt.title(f't-SNE Visualization of {name} Dataset')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title="Classes")
    plt.show()



if __name__ == "__main__":
    """
    Main execution block that processes multiple datasets, evaluates the DBSENCForest algorithm,
    and prints out the performance metrics.
    """
    dataset_list = []
    dataset_names = []

    datagen = Dataset1Generator()
    dataset_list.append(datagen.get_dataframe())
    dataset_names.append("Dataset1")

    datagen = Dataset2Generator()
    dataset_list.append(datagen.get_dataframe())
    dataset_names.append("Dataset2")

    datagen = Dataset3Generator()
    dataset_list.append(datagen.get_dataframe())
    dataset_names.append("Dataset3")

    datagen = Dataset4Generator()
    dataset_list.append(datagen.get_dataframe())
    dataset_names.append("Dataset4")

    data = pd.read_csv('data/wine.csv')
    columns = list(data.columns)
    columns[0], columns[-1] = columns[-1], columns[0]
    data = data[columns]
    dataset_list.append(data)
    dataset_names.append("Wine")
    perform_tsne_and_plot(data, "Wine")



    data = pd.read_csv('data/glass.csv')
    dataset_list.append(data)
    dataset_names.append("Glass")
    perform_tsne_and_plot(data, "Glass")

    data = pd.read_csv(
        'data/labeled_OES_data_2.csv')
    data.replace(',', '.', regex=True, inplace=True)
    dataset_list.append(data)
    dataset_names.append("OES_data")
    perform_tsne_and_plot(data, "OES_data")


    testa = DataManipulation()
    tests = ErrorMinimizationEvaluation()

    overall_results_FN = {}
    overall_results_FP = {}

    for n in range(1, len(dataset_list) + 1):
        for data, name in zip(dataset_list, dataset_names):
            data = tests.filter_rare_labels(data, 30)
            data[data.columns[-1]] = data[data.columns[-1]].astype(str)

            labels_set = set(data.iloc[:, -1].tolist())
            if n >= len(labels_set):
                continue

            all_combinations = list(itertools.combinations(labels_set, n))

            precision_sum_FN = 0
            recall_sum_FN = 0
            f1_sum_FN = 0
            adjusted_rand_sum_FN = 0

            precision_sum_FP = 0
            recall_sum_FP = 0
            f1_sum_FP = 0
            adjusted_rand_sum_FP = 0

            valid_combinations_FN = 0
            valid_combinations_FP = 0

            for pair in all_combinations:
                mod_data = data.copy()

                del_datas = []
                for label in pair:
                    mod_data, del_data = testa.delete_entries(mod_data, label, 0)
                    del_datas.append(del_data)

                del_data = del_datas[0]
                del_datas.pop(0)
                for del_data_next in del_datas:
                    del_data = pd.concat([del_data, del_data_next])

                del_data.reset_index(drop=True, inplace=True)
                del_data = del_data.sample(frac=1, random_state=42)

                print(f"Processing DataFrame: {name}, Combination: {pair}")

                try:
                    eval_sim_FN = tests.run_DBSENCForest(mod_data.copy(), del_data.copy(), "FN")
                    print(eval_sim_FN)
                    eval_sim_FP = tests.run_DBSENCForest(mod_data.copy(), del_data.copy(), "FP")
                    print(eval_sim_FP)
                except Exception as e:
                    print(f"Fehler: {e}")
                    continue

                precision_sum_FN += eval_sim_FN[0]
                recall_sum_FN += eval_sim_FN[1]
                f1_sum_FN += eval_sim_FN[2]
                adjusted_rand_sum_FN += eval_sim_FN[3]

                valid_combinations_FN += 1

                precision_sum_FP += eval_sim_FP[0]
                recall_sum_FP += eval_sim_FP[1]
                f1_sum_FP += eval_sim_FP[2]
                adjusted_rand_sum_FP += eval_sim_FP[3]

                valid_combinations_FP += 1

            if valid_combinations_FN > 0:
                if n not in overall_results_FN:
                    overall_results_FN[n] = {
                        "precision_sum": 0,
                        "recall_sum": 0,
                        "f1_sum": 0,
                        "adjusted_rand_sum": 0,
                        "count": 0,
                        "dataframes": []
                    }

                overall_results_FN[n]["precision_sum"] += precision_sum_FN
                overall_results_FN[n]["recall_sum"] += recall_sum_FN
                overall_results_FN[n]["f1_sum"] += f1_sum_FN
                overall_results_FN[n]["adjusted_rand_sum"] += adjusted_rand_sum_FN
                overall_results_FN[n]["count"] += valid_combinations_FN
                overall_results_FN[n]["dataframes"].append(name)

            if valid_combinations_FP > 0:
                if n not in overall_results_FP:
                    overall_results_FP[n] = {
                        "precision_sum": 0,
                        "recall_sum": 0,
                        "f1_sum": 0,
                        "adjusted_rand_sum": 0,
                        "count": 0,
                        "dataframes": []
                    }

                overall_results_FP[n]["precision_sum"] += precision_sum_FP
                overall_results_FP[n]["recall_sum"] += recall_sum_FP
                overall_results_FP[n]["f1_sum"] += f1_sum_FP
                overall_results_FP[n]["adjusted_rand_sum"] += adjusted_rand_sum_FP
                overall_results_FP[n]["count"] += valid_combinations_FP
                overall_results_FP[n]["dataframes"].append(name)

        print(f"\nFN Results for Combination Length {n}:")
        if n in overall_results_FN:
            precision_avg_FN = overall_results_FN[n]["precision_sum"] / overall_results_FN[n]["count"]
            recall_avg_FN = overall_results_FN[n]["recall_sum"] / overall_results_FN[n]["count"]
            f1_avg_FN = overall_results_FN[n]["f1_sum"] / overall_results_FN[n]["count"]
            adjusted_rand_avg_FN = overall_results_FN[n]["adjusted_rand_sum"] / overall_results_FN[n]["count"]

            print(f"Precision: {precision_avg_FN:.4f}, Recall: {recall_avg_FN:.4f}, F1: {f1_avg_FN:.4f}, Adjusted Rand Index: {adjusted_rand_avg_FN:.4f}")
            print(f"DataFrames used for Length {n}: {overall_results_FN[n]['dataframes']}")
        else:
            print("No valid combinations for this length.")

        print(f"\nFP Results for Combination Length {n}:")
        if n in overall_results_FP:
            precision_avg_FP = overall_results_FP[n]["precision_sum"] / overall_results_FP[n]["count"]
            recall_avg_FP = overall_results_FP[n]["recall_sum"] / overall_results_FP[n]["count"]
            f1_avg_FP = overall_results_FP[n]["f1_sum"] / overall_results_FP[n]["count"]
            adjusted_rand_avg_FP = overall_results_FP[n]["adjusted_rand_sum"] / overall_results_FP[n]["count"]

            print(f"Precision: {precision_avg_FP:.4f}, Recall: {recall_avg_FP:.4f}, F1: {f1_avg_FP:.4f}, Adjusted Rand Index: {adjusted_rand_avg_FP:.4f}")
            print(f"DataFrames used for Length {n}: {overall_results_FP[n]['dataframes']}")
        else:
            print("No valid combinations for this length.")