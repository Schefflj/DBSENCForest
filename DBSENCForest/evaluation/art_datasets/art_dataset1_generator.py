import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
import pandas as pd
import matplotlib.pyplot as plt

class Dataset1Generator:

    def get_dataframe(self):
        """
        Generates a synthetic dataset with varied densities, converts it to a pandas DataFrame,
        and visualizes the dataset with a scatter plot.

        :return: pandas DataFrame, containing the generated features and labels.
        """
        random_state = 42

        X1, y1 = make_blobs(n_samples=100, centers=[[200, 10]], cluster_std=[10], random_state=random_state)
        X2, y2 = make_blobs(n_samples=100, centers=[[10, 10]], cluster_std=[10], random_state=random_state)
        X3, y3 = make_blobs(n_samples=100, centers=[[10, 200]], cluster_std=[10], random_state=random_state)
        X4, y4 = make_blobs(n_samples=100, centers=[[50, 50]], cluster_std=[10], random_state=random_state)
        X5 = np.random.exponential(scale=200.0, size=(300, 2)) + [80, 80]
        y5 = np.full(300, 0)

        y1 = np.where(y1 == 0, 1, 1)
        y2 = np.where(y2 == 0, 2, 2)
        y3 = np.where(y3 == 0, 3, 3)
        y4 = np.where(y4 == 0, 4, 4)
        y5 = np.where(y5 == 0, 5, 5)

        X = np.vstack((X1, X2, X3, X4, X5))
        y = np.hstack((y1, y2, y3, y4, y5))

        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['label'] = y

        plt.figure(figsize=(5, 5))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.title("Art_Df1")
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")

        handles, labels = scatter.legend_elements(prop="colors")
        legend_labels = [f"{int(label)}" for label in np.unique(y)]
        plt.legend(handles, legend_labels, title="Labels", loc="upper right", bbox_to_anchor=(1, 1))

        plt.show()

        return df