from sklearn_extra.cluster import KMedoids
class KMedoids_methods():
    @staticmethod
    def kmedoids_clustering_with_sim_matrix(sim_matrix, n_clusters):
        """
        Performs K-Medoids clustering using a precomputed similarity (or dissimilarity) matrix.

        :param sim_matrix: list[list[float]], The precomputed similarity or dissimilarity matrix.
        :param n_clusters: int, The number of clusters to form.
        :return: list[int], The cluster labels for each point in the dataset.
        """

        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        clusters = kmedoids.fit(sim_matrix)

        return clusters.labels_

    @staticmethod
    def kmedoids_clustering(data, n_clusters):
        """
        Performs K-Medoids clustering directly on the dataset.

        :param data: list[list[float]], The dataset to cluster, where each element is a feature vector.
        :param n_clusters: int, The number of clusters to form.
        :return: list[int], The cluster labels for each point in the dataset.
        """
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
        clusters = kmedoids.fit(data)

        return clusters.labels_