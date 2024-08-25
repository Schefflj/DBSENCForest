import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class Minas:

    def __init__(self,
                 kini=3,
                 cluster_algorithm='kmeans',
                 random_state=0,
                 min_short_mem_trigger=10,
                 min_examples_cluster=10,
                 threshold_strategy=1,
                 threshold_factor=1.1,
                 window_size=100,
                 update_summary=False,
                 animation=False):
        """
        Initializes a MINAS object

        :param kini: int, Number of initial clusters per class.
        :param cluster_algorithm: str, Clustering algorithm to use ('kmeans' by default).
        :param random_state: int, Random seed for reproducibility.
        :param min_short_mem_trigger: int, Minimum number of instances in short-term memory to trigger novelty detection.
        :param min_examples_cluster: int, Minimum number of examples required for a cluster to be considered representative.
        :param threshold_strategy: int, Strategy to determine the cluster assignment threshold.
        :param threshold_factor: float, Factor to adjust the cluster assignment threshold.
        :param window_size: int, Number of samples after which old clusters are forgotten.
        :param update_summary: bool, Whether to dynamically update cluster summaries.
        :param animation: bool, Whether to generate animation frames of the clustering process.
        """
        self.kini = kini
        self.random_state = random_state
        self.cluster_algorithm = cluster_algorithm
        self.microclusters = []
        self.before_offline_phase = True
        self.short_mem = []
        self.sleep_mem = []
        self.min_short_mem_trigger = min_short_mem_trigger
        self.min_examples_cluster = min_examples_cluster
        self.threshold_strategy = threshold_strategy
        self.threshold_factor = threshold_factor
        self.window_size = window_size
        self.update_summary = update_summary
        self.animation = animation
        self.anomaly_mode = False
        self.sample_counter = 0

    def set_anomaly_mode(self, mode=True):
        """
        Sets the anomaly detection mode.

        :param mode: bool, If True, all instances will be treated as anomalies.
        """
        self.anomaly_mode = mode

    def fit(self, X, y, classes=None, sample_weight=None):
        """
        Performs the initial offline clustering on a labeled dataset.

        :param X: numpy array, Feature data for clustering.
        :param y: numpy array, Labels corresponding to X.
        :param classes: list, Optional list of all possible classes.
        :param sample_weight: array-like, Sample weights for clustering, if any.
        :return: self
        """
        self.microclusters = self.offline(X, y)
        self.before_offline_phase = False
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Performs online learning by updating clusters with new data points.

        :param X: numpy array, New feature data to be clustered.
        :param y: numpy array, Labels corresponding to X.
        :param classes: list, Optional list of all possible classes.
        :param sample_weight: array-like, Sample weights for clustering, if any.
        :return: self
        """
        self.sample_counter += 1
        if self.before_offline_phase:
            self.fit(X, y)
        else:
            y_preds, cluster_preds = self.predict(X, ret_cluster=True)
            timestamp = self.sample_counter
            for point_x, y_pred, cluster in zip(X, y_preds, cluster_preds):
                if y_pred != -1:
                    cluster.update_cluster(point_x, y_pred, timestamp, self.update_summary)
                else:
                    self.short_mem.append(ShortMemInstance(point_x, timestamp))
                    if len(self.short_mem) >= self.min_short_mem_trigger:
                        self.novelty_detect()
        return self

    def predict(self, X, ret_cluster=False):
        """
        Predicts the labels for the input data based on the closest microcluster.

        :param X: numpy array, Feature data for which labels are predicted.
        :param ret_cluster: bool, If True, also returns the corresponding clusters.
        :return: numpy array of predicted labels, and optionally a list of corresponding clusters.
        """
        pred_labels = []
        pred_clusters = []

        for point in X:
            if self.anomaly_mode:
                pred_labels.append(-1)
                pred_clusters.append(None)
            else:
                closest_cluster = min(self.microclusters,
                                      key=lambda cl: cl.distance_to_centroid(point))
                if closest_cluster.encompasses(point):
                    pred_labels.append(closest_cluster.label)
                    pred_clusters.append(closest_cluster)
                else:
                    pred_labels.append(-1)
                    pred_clusters.append(None)

        if ret_cluster:
            return np.asarray(pred_labels), pred_clusters
        else:
            return np.asarray(pred_labels)

    def predict_proba(self, X):
        # TODO: Implement predict_proba if necessary
        pass

    def offline(self, X_train, y_train):
        """
        Executes the offline phase by clustering the training data into microclusters.

        :param X_train: numpy array, Feature data used for offline clustering.
        :param y_train: numpy array, Labels corresponding to X_train.
        :return: list of MicroCluster objects
        """
        microclusters = []
        timestamp = len(X_train)
        if self.cluster_algorithm == 'kmeans':
            for y_class in np.unique(y_train):
                X_class = X_train[y_train == y_class]
                class_cluster_clf = KMeans(n_clusters=self.kini, random_state=self.random_state)
                class_cluster_clf.fit(X_class)

                for class_cluster in np.unique(class_cluster_clf.labels_):
                    cluster_instances = X_class[class_cluster_clf.labels_ == class_cluster]
                    microclusters.append(
                        MicroCluster(y_class, cluster_instances, timestamp)
                    )

        return microclusters

    def novelty_detect(self):
        """
        Detects and integrates novel clusters from instances in short-term memory.
        """
        possible_clusters = []
        X = np.array([instance.point for instance in self.short_mem])
        if self.cluster_algorithm == 'kmeans':
            cluster_clf = KMeans(n_clusters=self.kini, random_state=self.random_state)
            cluster_clf.fit(X)
            for cluster_label in np.unique(cluster_clf.labels_):
                cluster_instances = X[cluster_clf.labels_ == cluster_label]
                possible_clusters.append(
                    MicroCluster(-1, cluster_instances, self.sample_counter))
            for cluster in possible_clusters:
                if cluster.is_cohesive(self.microclusters) and cluster.is_representative(self.min_examples_cluster):
                    closest_cluster = cluster.find_closest_cluster(self.microclusters)
                    closest_distance = cluster.distance_to_centroid(closest_cluster.centroid)

                    threshold = self.best_threshold(cluster, closest_cluster,
                                                    self.threshold_strategy, self.threshold_factor)

                    if closest_distance < threshold:
                        cluster.label = closest_cluster.label
                    elif self.sleep_mem:
                        closest_cluster = cluster.find_closest_cluster(self.sleep_mem)
                        closest_distance = cluster.distance_to_centroid(closest_cluster.centroid)
                        if closest_distance < threshold:
                            cluster.label = closest_cluster.label
                            self.sleep_mem.remove(closest_cluster)
                            closest_cluster.timestamp = self.sample_counter
                            self.microclusters.append(closest_cluster)
                        else:
                            cluster.label = max([cluster.label for cluster in self.microclusters]) + 1
                    else:
                        cluster.label = max([cluster.label for cluster in self.microclusters]) + 1

                    self.microclusters.append(cluster)
                    for instance in cluster.instances:
                        self.short_mem.remove(instance)

    def best_threshold(self, new_cluster, closest_cluster, strategy, factor):
        """
        Determines the best threshold for assigning a new cluster to an existing class.

        :param new_cluster: MicroCluster, The new cluster to be classified.
        :param closest_cluster: MicroCluster, The closest existing cluster.
        :param strategy: int, Strategy to determine the threshold.
        :param factor: float, Factor to adjust the threshold.
        :return: float, The calculated threshold value.
        """
        def run_strategy_1():
            return factor * np.std(closest_cluster.distance_to_centroid(closest_cluster.instances))

        if strategy == 1:
            return run_strategy_1()
        else:
            factor_2 = factor_3 = factor
            clusters_same_class = self.get_clusters_in_class(closest_cluster.label)
            if len(clusters_same_class) == 1:
                return run_strategy_1()
            else:
                class_centroids = np.array([cluster.centroid for cluster in clusters_same_class])
                distances = closest_cluster.distance_to_centroid(class_centroids)
                if strategy == 2:
                    return factor_2 * np.max(distances)
                elif strategy == 3:
                    return factor_3 * np.mean(distances)

    def get_clusters_in_class(self, label):
        """
        Retrieves all clusters corresponding to a specific class label.

        :param label: int, The label of the class.
        :return: list of MicroCluster objects
        """
        return [cluster for cluster in self.microclusters if cluster.label == label]

    def trigger_forget(self):
        """
        Moves outdated clusters to sleep memory and removes old instances from short-term memory.
        """
        for cluster in self.microclusters:
            if cluster.timestamp < self.sample_counter - self.window_size:
                self.sleep_mem.append(cluster)
                self.microclusters.remove(cluster)
        for instance in self.short_mem:
            if instance.timestamp < self.sample_counter - self.window_size:
                self.short_mem.remove(instance)

    def plot_clusters(self):
        """
        Visualizes the current state of clustering, including microclusters and short-term memory instances.
        """
        points = pd.DataFrame(columns=['x', 'y', 'pred_label'])
        cluster_info = pd.DataFrame(columns=['label', 'centroid', 'radius'])
        for cluster in self.microclusters:
            cluster_info = cluster_info.append(pd.Series({'label': cluster.label,
                                                          'centroid': cluster.centroid,
                                                          'radius': cluster.radius}),
                                               ignore_index=True)
            for point in cluster.instances:
                points = points.append(pd.Series({'x': point[0],
                                                  'y': point[1],
                                                  'pred_label': cluster.label}),
                                       ignore_index=True)
        for mem_instance in self.short_mem:
            points = points.append(pd.Series({'x': mem_instance.point[0],
                                              'y': mem_instance.point[1],
                                              'pred_label': -1}),
                                   ignore_index=True)

        color_names = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        assert len(cluster_info.label.unique()) <= len(color_names)
        colormap = pd.DataFrame({'name': color_names}, index=range(-1, len(color_names) - 1))
        mapped_label_colors = colormap.loc[points['pred_label']].values[:, 0]
        plt.scatter(points['x'], points['y'], c=mapped_label_colors, s=10, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')

        circles = []
        for label, centroid, radius in cluster_info.values:
            circles.append(plt.Circle((centroid[0], centroid[1]), radius,
                                      color=colormap.loc[label].values[0], alpha=0.1))
        for circle in circles:
            plt.gcf().gca().add_artist(circle)

        import os
        if not os.path.exists('animation'):
            os.makedirs('animation')
        plt.savefig(f'animation/clusters_{self.animation_frame_num:05}.png', dpi=300)
        plt.close()
        self.animation_frame_num += 1

    def confusion_matrix(self, X_test=None, y_test=None):
        """
        Generates a confusion matrix comparing predicted cluster labels with true labels.

        :param X_test: numpy array, Test feature data.
        :param y_test: numpy array, True labels for the test data.
        :return: pandas DataFrame, Confusion matrix.
        """
        y_test_classes = np.unique(y_test)
        detected_classes = np.unique([cluster.label for cluster in self.microclusters])
        conf_matrix = pd.DataFrame(np.zeros((len(y_test_classes), len(detected_classes) + 1), dtype=np.int32))
        conf_matrix.index = y_test_classes
        conf_matrix.rename(columns={len(detected_classes): -1}, inplace=True)

        y_preds = self.predict(X_test)

        for y_true, y_pred in zip(y_test, y_preds):
            conf_matrix.loc[y_true, y_pred] += 1

        return conf_matrix

class MicroCluster:

    def __init__(self, label, instances, timestamp=0):
        """
        Initializes a microcluster with a given label and instances.

        :param label: int, The label of the cluster.
        :param instances: numpy array, The instances that form the cluster.
        :param timestamp: int, Timestamp of the cluster creation or last update.
        """
        self.label = label
        self.instances = instances
        self.n = len(instances)
        self.linear_sum = instances.sum(axis=0)
        self.centroid = self.linear_sum / self.n
        self.squared_sum = np.square(np.linalg.norm(self.instances, axis=1)).sum()
        self.timestamp = timestamp
        self.update_properties()

    def __str__(self):
        """
        Returns a string representation of the microcluster, including its properties.

        :return: str, String describing the microcluster.
        """
        return f"""Target class {self.label}
# of instances: {self.n}
Linear sum: {self.linear_sum}
Squared sum: {self.squared_sum}
Centroid: {self.centroid}
Radius: {self.radius}
Timestamp of last change: {self.timestamp}"""

    def get_radius(self):
        """
        Calculates the radius of the cluster based on the variance of the instances.

        :return: float, The calculated radius.
        """
        factor = 1.5
        diff = (self.squared_sum / self.n) - np.dot(self.centroid, self.centroid)
        if diff > 1e-15:
            return factor * np.sqrt(diff)
        else:
            return 0

    def distance_to_centroid(self, X):
        """
        Calculates the distance from an input point or set of points to the cluster centroid.

        :param X: numpy array, A point or set of points.
        :return: float or numpy array, The distance(s) to the centroid.
        """
        if len(X.shape) == 1:
            return np.linalg.norm(X - self.centroid)
        else:
            return np.linalg.norm(X - self.centroid, axis=1)

    def encompasses(self, X):
        """
        Checks whether the input point or set of points falls within the cluster's radius.

        :param X: numpy array, A point or set of points.
        :return: bool, Whether the point(s) are within the cluster.
        """
        return np.less(self.distance_to_centroid(X), self.radius)

    def find_closest_cluster(self, clusters):
        """
        Finds the closest cluster to this microcluster among a list of clusters.

        :param clusters: list of MicroCluster objects, The clusters to compare against.
        :return: MicroCluster, The closest cluster.
        """
        return min(clusters, key=lambda cl: cl.distance_to_centroid(self.centroid))

    def update_cluster(self, X, y, timestamp, update_summary):
        """
        Updates the microcluster with a new instance.

        :param X: numpy array, The new instance to add.
        :param y: int, The label of the new instance.
        :param timestamp: int, The current timestamp.
        :param update_summary: bool, Whether to update the summary statistics of the cluster.
        """
        assert len(X.shape) == 1
        self.timestamp = timestamp
        self.instances = np.append(self.instances, [X], axis=0)
        if update_summary:
            self.n += 1
            self.linear_sum = np.sum([self.linear_sum, X], axis=0)
            self.squared_sum = np.sum([self.squared_sum, np.square(X)], axis=0)
            self.update_properties()

    def update_properties(self):
        """
        Updates the centroid and radius of the cluster based on its current instances.
        """
        self.centroid = self.linear_sum / self.n
        self.radius = self.get_radius()

    def is_cohesive(self, clusters):
        """
        Determines if the cluster is cohesive relative to other clusters.

        :param clusters: list of MicroCluster objects, The clusters to compare against.
        :return: bool, Whether the cluster is cohesive.
        """
        b = self.distance_to_centroid(self.find_closest_cluster(clusters).centroid)
        a = np.std(self.distance_to_centroid(self.instances))
        silhouette = (b - a) / max(a, b)
        return silhouette > 0

    def is_representative(self, min_examples):
        """
        Checks if the cluster has enough instances to be representative.

        :param min_examples: int, The minimum number of instances required.
        :return: bool, Whether the cluster is representative.
        """
        return self.n >= min_examples

class ShortMemInstance:
    def __init__(self, point, timestamp):
        """
        Initializes a short-term memory instance.

        :param point: numpy array, The feature data of the instance.
        :param timestamp: int, The timestamp when the instance was added to short-term memory.
        """
        self.point = point
        self.timestamp = timestamp

    def __eq__(self, other):
        """
        Checks if two instances are equal based on their feature data.

        :param other: ShortMemInstance or numpy array, The instance to compare against.
        :return: bool, Whether the instances are equal.
        """
        if type(other) == np.ndarray:
            return np.all(self.point == other)