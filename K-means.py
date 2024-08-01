# imports
import math
import random
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# Function computing the distance between two data points using math.sqrt
def compute_dist(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    )

# Function reads data in from the csv files using a for loop
def read_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Removing irrelevant text inside parentheses and extracting
        # relevant column names by using .strip text within brackets
        def clean_column_name(column_name):            
            return re.sub(r'\(.*\)', '', column_name).strip()

        # Using a for loop to return the used columns
        columns = {clean_column_name(col): col for col in reader.fieldnames}
        for row in reader:
            countries = row[columns['Countries']]
            birth_rate = float(row[columns['BirthRate']])
            life_expectancy = float(row[columns['LifeExpectancy']])
            data.append((countries, birth_rate, life_expectancy))
    return data

# np.argmin used to find the smallest element of the closest centroid to
# each point out of all the centroids and returning
def find_closest_centroid(point, centroids):
    distances = [
        compute_dist((point[1], point[2]), centroid) for centroid in centroids
    ]
    closest_centroid_index = np.argmin(distances)
    return closest_centroid_index


# Using a for loop function to visualize the clusters
# Extracting the coordinates of the point
# Finding the closest centroid index
# Appending the point to the appropriate cluster
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        point_coordinates = (point[1], point[2])
        closest_centroid_index = find_closest_centroid(point, centroids)
        clusters[closest_centroid_index].append(point)
    return clusters

# Creating the cluster labels and colors used
# Creating the cluster labels, assigning colors
def plot_clusters(clusters, centroids):
    cluster_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        birth_rates = [point[1] for point in cluster]
        life_expectancies = [point[2] for point in cluster]
        plt.scatter(
            birth_rates,
            life_expectancies,
            c=cluster_colors[i % len(cluster_colors)],
            label=f'Cluster {i + 1}'
        )

    # Plotting centroids
    centroid_colors = [
        'black', 'orange', 'purple', 'brown', 'pink', 'grey', 'olive'
    ]
    for i, centroid in enumerate(centroids):
        plt.scatter(
            centroid[0], centroid[1],
            c=centroid_colors[i % len(centroid_colors)],
            marker='x', s=200, linewidths=3
        )

    # Labels for the x and y axis of the k-means plotter graph
    plt.xlabel('Birth Rate')
    plt.ylabel('Life Expectancy')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()

# Writing the initialization procedure, taking a random sample for the centroids
def initialize_centroids(data, k):
    random_indices = random.sample(range(len(data)), k)
    centroids = [(data[i][1], data[i][2]) for i in random_indices]
    return centroids

# Using a for loop to find the mean of the birth rate and life expectancy
# both points are appended to create the dot values on the graph
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        mean_birth_rate = np.mean([point[1] for point in cluster])
        mean_life_expectancy = np.mean([point[2] for point in cluster])
        new_centroids.append((mean_birth_rate, mean_life_expectancy))
    return new_centroids

# Initializing the centroids
def kmeans(data, k, iterations):
    centroids = initialize_centroids(data, k)

    # Iterate for the specified number of iterations
    # Assign clusters based on the current centroids
    for iteration in range(iterations):
        clusters = assign_clusters(data, centroids)
        total_squared_distance = 0
        for i, cluster in enumerate(clusters):
            centroid = centroids[i]
            squared_distances = [
                compute_dist((point[1], point[2]), centroid)**2
                for point in cluster
            ]
            total_squared_distance += sum(squared_distances)

        print(
            f"\nIteration {iteration + 1}:\nSum of squared distances = "
            f"{total_squared_distance:.2f}"
        )
        # Update centroids based on the new clusters
        centroids = update_centroids(clusters)

    return clusters, centroids
    
# Writing the observations further down the code
def write_observations(filename, observations):
    with open(filename, 'w') as file:
        file.write(observations)

# Reading the data
# .csv data filepaths
def main():  
    datasets = [  
        {'filename': r'data2008.csv'},
        {'filename': r'data1953.csv'},
        {'filename': r'dataBoth.csv'},
    ]
    observations = ""

    # Getting user input for number of iterations and number of clusters to be used
    iterations = int(input("\nEnter the number of iterations for K-Means: "))
    k = int(input("Enter the number of clusters for K-Means: "))

    # Read the data files
    for dataset in datasets:  
        filename = dataset['filename']
        observations += f"\nRunning K-Means on {filename} with k={k} clusters\n"
        data = read_data(filename)
        clusters, centroids = kmeans(data, k, iterations)
        plot_clusters(clusters, centroids)

        # Outputting the cluster information
        for i, cluster in enumerate(clusters):
            observations += f"Cluster {i + 1}: {len(cluster)} countries\n"
            for country in cluster:
                observations += f" {country[0]}\n"
            observations += "\n"

        # Calculating and printing the mean Life Expectancy and Birth Rate for each cluster
        for i, centroid in enumerate(centroids):
            observations += (
                f"Cluster {i + 1} Centroid: \nBirth Rate = {centroid[0]:.2f}, "
                f"\nLife Expectancy = {centroid[1]:.2f}\n\n"
            )

    write_observations(r'k-means output.txt', observations)

if __name__ == "__main__":
    main()
