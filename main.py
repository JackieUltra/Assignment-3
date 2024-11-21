import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
from sklearn.metrics import  silhouette_score
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    kmeans = KMeans(k=3,metric='euclidean', max_iter=300, tol=0.0001)
    
    # Fit model
    kmeans.fit(og_iris)
    
    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    predictions = kmeans.predict(df)
    
    # You can choose which scoring method you'd like to use here:
    score = silhouette_score(df, predictions)
    print("Silhouette Score:", score)
    
    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(df, predictions, kmeans, score)
    
    # Try different numbers of clusters
    inertia_values = []
    k_values = range(1, 10)
    for k in k_values:
        kmeans_test = KMeans(k=k, metric='euclidean', max_iter=300, tol=0.0001)
        kmeans_test.fit(og_iris)
        inertia_values.append(kmeans_test.get_error())
    
    # Plot the elbow plot
    plt.figure()
    plt.title('Elbow Method to Optimize k')
    plt.plot(k_values, inertia_values, marker='x')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()
    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 3
    
    Reasoning: 
    There are most likely 3 distinct groups we can see this from both the graph and the silouette score.
    The intertia slows down signficantly by the 3rd k group. the 2nd group looks solid too but the score has 
    not sufficently leveled out yet. Also a Silouette score of 0.44 is pretty solid.
    """

    
if __name__ == "__main__":
    main()