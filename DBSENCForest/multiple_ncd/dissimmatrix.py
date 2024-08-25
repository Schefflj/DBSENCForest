
class Dissimmatrix:


    def initialize_matrix(self, n):
        """
        Initializes an n x n matrix filled with zeros.

        :param n: int, The size of the matrix (n x n).
        :return: list[list[int]], The initialized matrix.
        """
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        return matrix

    def count_matches(self, list1, list2):
        """
        Counts the number of matches between two lists up to the length of the shorter list.

        :param list1: list[int], The first list of node IDs.
        :param list2: list[int], The second list of node IDs.
        :return: int, The count of matching elements at corresponding positions in the two lists.
        """
        min_length = min(len(list1), len(list2))

        count = 0
        for i in range(min_length):
            if list1[i] == list2[i]:
                count += 1

        return count

    def calculate_similarity_matrix(self, node_id_lists):
        """
        Calculates a similarity matrix for a given list of node ID sequences.

        :param node_id_lists: list[list[int]], A list where each element is a sequence of node IDs for a data point.
        :return: list[list[float]], The similarity matrix, where each element is the similarity score between two sequences.
        """

        n = len(node_id_lists)

        similarity_matrix = self.initialize_matrix(n)

        i = 0


        while i < n:
            j = i + 1
            while j < n:

                similarity_matrix[i][j] = 1 - (self.count_matches(node_id_lists[i], node_id_lists[j]) / len(node_id_lists[0]))
                similarity_matrix[j][i] = similarity_matrix[i][j]


                j += 1
            i += 1

        return similarity_matrix











