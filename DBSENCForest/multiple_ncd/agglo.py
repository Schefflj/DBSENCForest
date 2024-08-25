from sklearn.cluster import AgglomerativeClustering


class HierarchicalClusteringMethods():

    @staticmethod
    def hierarchical_clustering_with_sim_matrix(sim_matrix, n_clusters):
        """
        Perform hierarchical clustering using a similarity matrix.

        Parameters:
        - sim_matrix: array-like of shape (n_samples, n_samples)
                      The precomputed similarity matrix.
        - n_clusters: int
                      The number of clusters to find.

        Returns:
        - labels: array of shape (n_samples,)
                  Cluster labels for each point.
        """
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        clusters = clustering.fit(sim_matrix)

        return clusters.labels_

    @staticmethod
    def hierarchical_clustering(data, n_clusters):
        """
        Perform hierarchical clustering on the data.

        Parameters:
        - data: array-like of shape (n_samples, n_features)
                The input data.
        - n_clusters: int
                      The number of clusters to find.

        Returns:
        - labels: array of shape (n_samples,)
                  Cluster labels for each point.
        """
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit(data)

        return clusters.labels_