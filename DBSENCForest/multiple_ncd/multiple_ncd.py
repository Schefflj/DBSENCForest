import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from DBSENCForest.data_structure.ext_dataframe import ExtDataFrame
from DBSENCForest.data_structure.senc_forest.senc_forests_algorithm import SencForestsAlgorithm
from DBSENCForest.multiple_ncd.dbscan import DBSCAN_methods
from DBSENCForest.multiple_ncd.dissimmatrix import Dissimmatrix


class Multiple_NCD:

    def __init__(self, sim_matrix_given, ext_training_data, new_class_buffer_df, initial_training_df, data_stream_classifier, minimize_error):
        """
        Initializes the Multiple Novel Class Detection (NCD) process.

        :param sim_matrix_given: Boolean indicating if the similarity matrix is precomputed.
        :param ext_training_data: Extended training data used for the initial model.
        :param new_class_buffer_df: DataFrame containing potential new class samples.
        :param initial_training_df: DataFrame containing the initial training data.
        :param data_stream_classifier: The classifier handling the data stream.
        :param minimize_error: Strategy for error minimization ('FP' for false positives, 'FN' for false negatives).
        """
        self.sim_matrix_given = sim_matrix_given
        self.ext_training_data = ext_training_data
        self.new_class_buffer_df = new_class_buffer_df
        self.initial_training_df = initial_training_df
        self.data_stream_classifier = data_stream_classifier
        self.minimize_error = minimize_error

    def get_answer_masses(self, ext_df, forest):
        """
        Computes the answer masses for the given data using the specified forest.

        :param ext_df: Extended DataFrame containing samples to evaluate.
        :param forest: The forest model used for estimating answer masses.
        :return: List of masses calculated for each entry.
        """
        masses_list = []

        for entry in ext_df.sample_dict.values():

            entry = entry.row.iloc[:-1]

            entry = entry.apply(pd.to_numeric, errors='coerce').to_numpy()

            senc_algorithm = SencForestsAlgorithm()

            mass, mtimetest = senc_algorithm.sence_estimation(entry.reshape(1, -1), forest, 1,
                                                                   forest.anomaly)

            masses_list.append(mass)

        return masses_list




    def create_node_id_lists(self, masses):
        """
        Creates lists of node IDs from the calculated masses.

        :param masses: List of masses obtained from the forest model.
        :return: List of node ID lists corresponding to each sample.
        """

        node_id_lists = []

        for mass in masses:

            node_ids = mass[:, 3].copy()

            node_id_lists.append(node_ids)

        return node_id_lists

    def find_anomalies(self, masses):
        """
        Identifies anomalies within the masses.

        :param masses: List of masses obtained from the forest model.
        :return: List of indices corresponding to detected anomalies.
        """

        anomaly_list = []

        i = 0
        for mass in masses:
            anomaly_vals = mass[:, 0].copy()
            senc_algorithm = SencForestsAlgorithm()
            scores = senc_algorithm.tabulate(anomaly_vals)
            score_1 = scores[scores[:, 1] == np.max(scores[:, 1]), :]
            if score_1[0][0] == 1:
                anomaly_list.append(i)

            i += 1

        return anomaly_list

    def calculate_silhouette_score(self, sim_matrix, cluster_labels, cluster, df):
        """
        Calculates the silhouette score for a given cluster.

        :param sim_matrix: Precomputed similarity matrix (if available).
        :param cluster_labels: Cluster labels assigned to each sample.
        :param cluster: The cluster for which the silhouette score is calculated.
        :param df: DataFrame used for silhouette score calculation if the similarity matrix is not provided.
        :return: The silhouette score for the specified cluster.
        """

        if self.sim_matrix_given:
            silhouette_vals = silhouette_samples(sim_matrix, cluster_labels, metric='precomputed')
        else:
            silhouette_vals = silhouette_samples(df, cluster_labels)

        cluster_silhouette_vals = silhouette_vals[cluster_labels == cluster]
        return np.mean(cluster_silhouette_vals)



    def check_isolation(self, cluster_samples, known_size, tolerance_per):
        """
        Checks if a cluster is isolated based on the given parameters.

        :param cluster_samples: Indices of the samples within the cluster.
        :param known_size: Number of known samples.
        :param tolerance_per: Tolerance percentage for outliers.
        :return: Tuple indicating whether the cluster is isolated and the number of outliers.
        """

        new_samples_in_cluster = 0

        tolerance_counter = 0

        for sample in cluster_samples:

            if sample < known_size:
                tolerance_counter += 1

            if sample >= known_size:
                new_samples_in_cluster += 1


        if tolerance_counter / (new_samples_in_cluster + tolerance_counter) > tolerance_per:
            return False, 0

        return True, tolerance_counter

    def find_isolated_clusters_with_params(self, eps, min_samples, min_cluster_size, df_known, df_unknown, sim_matrix, anomalies):
        """
        Identifies isolated clusters using DBSCAN with the specified parameters.

        :param eps: Epsilon parameter for DBSCAN.
        :param min_samples: Minimum number of samples to form a cluster.
        :param min_cluster_size: Minimum size of a cluster to be considered isolated.
        :param df_known: DataFrame containing known samples.
        :param df_unknown: DataFrame containing unknown samples.
        :param sim_matrix: Precomputed similarity matrix (if available).
        :param anomalies: List of detected anomalies.
        :return: Tuple containing information about the isolated clusters.
        """
        all_df = None
        if self.sim_matrix_given:
            clusters_combined = DBSCAN_methods.dbscan_clustering_with_sim_matrix(sim_matrix, eps, min_samples)

        else:
            all_df = pd.concat([df_known, df_unknown])
            all_df = all_df.drop(all_df.columns[-1], axis=1)
            clusters_combined = DBSCAN_methods.dbscan_clustering(all_df, eps, min_samples)



        known_size = len(df_known)

        cluster_counts = pd.Series(clusters_combined).value_counts()

        isolated_clusters = []
        isolation_counter = 0
        silhouette_vals = 0
        for cluster_label in cluster_counts.index:
            if cluster_label == -1:
                continue
            cluster_samples = np.where(clusters_combined == cluster_label)[0]
            isolated, outlier_counter = self.check_isolation(cluster_samples, known_size, len(anomalies) / len(sim_matrix))
            min_sized = cluster_counts[cluster_label] - outlier_counter >= min_cluster_size
            if isolated and min_sized:
                if len(cluster_counts) > 1:
                    silhouette_vals += self.calculate_silhouette_score(sim_matrix, clusters_combined, cluster_label, all_df)
                isolated_clusters.append(cluster_label)
                isolation_counter += cluster_counts[cluster_label]

        if isolated_clusters:
            return True, clusters_combined, set(isolated_clusters), isolation_counter, silhouette_vals / len(set(isolated_clusters))
        else:
            return False, clusters_combined, set(), isolation_counter, 0

    def multiple_novel_class_detection(self):
        """
        Executes the process of detecting multiple novel classes in the data stream.

        :return: DataFrame with updated labels after novel class detection.
        """

        df = self.ext_training_data.dataframe.copy()

        pseudo_class_samples_df = df[df[df.columns[-1]].str.contains("pseudo_class", na=False)]
        new_pseudo_class_samples_df = self.new_class_buffer_df.dataframe

        all_nc_df = pd.concat([new_pseudo_class_samples_df, pseudo_class_samples_df])
        all_nc_df.iloc[:, -1] = -2
        all_nc_df = all_nc_df.reset_index(drop=True)
        ext_all_nc_df = ExtDataFrame(all_nc_df)

        all_df = pd.concat([self.initial_training_df, all_nc_df]).reset_index(drop=True)
        ext_all_df = ExtDataFrame(all_df)

        sim_matrix = [0]

        anomalies = [0]

        if self.sim_matrix_given:

            ncd_forest = self.data_stream_classifier.recreate_forest(ext_all_df.dataframe, {})

            masses = self.get_answer_masses(ext_all_df, ncd_forest)

            node_id_lists = self.create_node_id_lists(masses)

            clustering = Dissimmatrix()

            sim_matrix = clustering.calculate_similarity_matrix(node_id_lists)

            anomalies = self.find_anomalies(masses)

        best_clustering = (False, [], {}, 0, 0)

        if self.minimize_error == "FP":
            start_eps = 0.01
            start_min_pts = 3 * all_nc_df.shape[1]
            eps_steps = np.arange(start_eps, 1, 0.01)
            min_pts_steps = range(start_min_pts, 1, -1)

        if self.minimize_error == "FN":
            start_eps = 1
            start_min_pts = 2
            eps_steps = np.arange(start_eps, 0.01, -0.01)
            min_pts_steps = range(start_min_pts, 3 * all_nc_df.shape[1] + 1, 1)



        for min_samples in min_pts_steps:
            for eps in eps_steps:

                found, cluster_list, isolated_clusters, isolation_counter, silhouette_val = self.find_isolated_clusters_with_params(eps, min_samples, self.data_stream_classifier.sub_set_size, self.initial_training_df, all_nc_df, sim_matrix, anomalies)

                if not best_clustering[0]:
                    best_clustering = (found, cluster_list, isolated_clusters, isolation_counter, silhouette_val)

                if self.minimize_error == "FN" and found:

                    if isolation_counter >= best_clustering[3]:

                        if isolation_counter > best_clustering[3]:
                            best_clustering = (found, cluster_list, isolated_clusters, isolation_counter, silhouette_val)

                        if len(isolated_clusters) < len(best_clustering[2]):
                            best_clustering = (found, cluster_list, isolated_clusters, isolation_counter, silhouette_val)

                        if len(isolated_clusters) == len(best_clustering[2]) and silhouette_val > best_clustering[4]:
                            best_clustering = (found, cluster_list, isolated_clusters, isolation_counter, silhouette_val)


                if self.minimize_error == "FP" and found:

                    if len(isolated_clusters) >= len(best_clustering[2]):

                        if len(isolated_clusters) > len(best_clustering[2]):
                            best_clustering = (found, cluster_list, isolated_clusters, isolation_counter, silhouette_val)

                        if isolation_counter < best_clustering[3]:
                            best_clustering = (found, cluster_list, isolated_clusters, isolation_counter, silhouette_val)

                        if isolation_counter == best_clustering[3] and silhouette_val > best_clustering[4]:
                            best_clustering = (found, cluster_list, isolated_clusters, isolation_counter, silhouette_val)



            if self.minimize_error == "FN" and best_clustering[3] / len(all_nc_df) > 0.99:
                break

            if self.minimize_error == "FP" and (len(best_clustering[2]) * self.data_stream_classifier.sub_set_size) / len(all_nc_df) > 0.99:
                break

        cluster_list = best_clustering[1][len(self.initial_training_df):]
        low_quality_clusters = set(cluster_list) - best_clustering[2]
        low_quality_clusters.add(-1)
        updated_cluster_list = [-2 if item in low_quality_clusters else item for item in cluster_list]
        cluster_list_pseudos = ["pseudo_class" + str(value) for value in updated_cluster_list]
        updated_cluster_list_pseudos = [-2 if item in ["pseudo_class-2"] else item for item in cluster_list_pseudos]
        ext_all_nc_df.dataframe[ext_all_nc_df.dataframe.columns[-1]] = updated_cluster_list_pseudos
        i = 0
        k = 0
        buffer_size = len(self.data_stream_classifier.new_class_buffer)
        for entry in updated_cluster_list_pseudos:
            if k >= buffer_size:
                break
            if entry != -2:
                self.data_stream_classifier.new_class_buffer.pop(i)
            else:
                i += 1
            k += 1
        self.new_class_buffer_df = ExtDataFrame(self.ext_training_data.dataframe.copy().iloc[0:0])
        df = ext_all_nc_df.dataframe.copy()
        ext_all_nc_df.dataframe = df[df[df.columns[-1]] != -2]
        ext_update_buffer_df = ExtDataFrame(df[df[df.columns[-1]] == -2])
        self.new_class_buffer = []
        for entry in ext_update_buffer_df.sample_dict.values():
            self.new_class_buffer.append((entry, "empty", -2))

        return ext_all_nc_df.dataframe




