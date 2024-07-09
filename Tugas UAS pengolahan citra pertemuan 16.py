import cv2
import numpy as np
import random

def initialize_centroids(image, k):
    centroids = []
    for _ in range(k):
        i = random.randint(0, image.shape[0] - 1)
        j = random.randint(0, image.shape[1] - 1)
        centroids.append(image[i, j])
    return np.array(centroids, dtype=np.float32)

def assign_clusters(image, centroids):
    clusters = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            distances = np.linalg.norm(image[i, j] - centroids, axis=1)
            clusters[i, j] = np.argmin(distances)
    return clusters

def update_centroids(image, clusters, k):
    new_centroids = []
    for c in range(k):
        points = image[clusters == c]
        if len(points) > 0:
            new_centroids.append(np.mean(points, axis=0))
        else:
            new_centroids.append(np.zeros((3,)))
    return np.array(new_centroids, dtype=np.float32)

def kmeans_segmentation(image, k, max_iter=100):
    centroids = initialize_centroids(image, k)
    for _ in range(max_iter):
        clusters = assign_clusters(image, centroids)
        new_centroids = update_centroids(image, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters

# Load image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform K-means segmentation
k = 3
clusters = kmeans_segmentation(image, k)

# Create segmented image
segmented_image = np.zeros_like(image)
for i in range(k):
    segmented_image[clusters == i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Save segmented image
cv2.imwrite('segmented_image.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
