from sklearn.cluster import DBSCAN


class DBSCAN_methods():

    @staticmethod
    def dbscan_clustering_with_sim_matrix(sim_matrix, eps, min_samples):
        """
        Performs DBSCAN clustering using a precomputed similarity matrix.

        :param sim_matrix: np.ndarray, Precomputed similarity matrix (square matrix where each element represents the distance between points).
        :param eps: float, The maximum distance between two samples for them to be considered as in the same neighborhood.
        :param min_samples: int, The number of samples in a neighborhood for a point to be considered as a core point.
        :return: np.ndarray, The labels assigned to each data point after clustering.
        """

        db = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)

        clusters = db.fit(sim_matrix)

        return clusters.labels_

    @staticmethod
    def dbscan_clustering(data, eps, min_samples):
        """
        Performs DBSCAN clustering directly on the data.

        :param data: np.ndarray, Data to be clustered (features of the samples).
        :param eps: float, The maximum distance between two samples for them to be considered as in the same neighborhood.
        :param min_samples: int, The number of samples in a neighborhood for a point to be considered as a core point.
        :return: np.ndarray, The labels assigned to each data point after clustering.
        """

        db = DBSCAN(eps=eps, min_samples=min_samples)

        clusters = db.fit(data)

        return clusters.labels_