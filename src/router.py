import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

def get_binary_path(depth, i):
    '''
    Based on the embedding vector that the input image maps to, output
    a corresponding path in the graph
    '''
    # Convert `i` to binary and pad to `depth` bits
    binary_representation = bin(i)[2:].zfill(depth-1)
    
    # Convert the binary string to a list of integers
    binary_list = [int(bit) for bit in binary_representation]
    
    # Convert to a PyTorch tensor
    return torch.tensor(binary_list, dtype=torch.int32)

def map_emb_to_path(embs, depth):
    '''
    Given a list of embedding vectors, we perform k-means clustering hierarchically w/ k=2.
    Then we return a map from each emb vector to the corresponding path.
    '''
    cluster_indices = hierarchical_kmeans(embs, depth)
    return [get_binary_path(depth, cluster_indices[i]) for i in range(len(embs))]

def hierarchical_kmeans(X, depth, current_level=0, cluster_indices=None):
    """
    Hierarchical K-Means Clustering for k=2, returning cluster indices.
    
    X: torch tensor of shape [n, d] (n points, each of dimension d)
    depth: int, maximum depth of the hierarchy (depth - 1 iterations of clustering)
    current_level: int, the current depth in the hierarchy
    cluster_indices: torch tensor of shape [n], stores cluster indices of points
    
    Returns:
        cluster_indices: torch tensor of shape [n], the cluster index of each point
    """
    if cluster_indices is None:
        # Initialize cluster indices as 0 for all points
        cluster_indices = torch.zeros(X.shape[0], dtype=torch.int32)

    if current_level == depth - 1 or X.shape[0] <= 1:
        # Return the current cluster indices if depth is reached or only one point
        return cluster_indices

    # Convert PyTorch tensor to NumPy array for sklearn KMeans compatibility
    X_np = X.numpy()

    # Apply K-Means with k=2
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_np)

    # Assign points to clusters
    labels = torch.tensor(kmeans.labels_, dtype=torch.int32)

    # Split data based on k-means labels
    cluster_1 = X[labels == 0]
    cluster_2 = X[labels == 1]

    points_to_move = max(len(cluster_1), len(cluster_2)) - 2**(depth-current_level-2)
    if (points_to_move > 0):
        centers = kmeans.cluster_centers_
        bigger_cluster = 0 if len(cluster_1) > len(cluster_2) else 1
        dist = np.sum((X_np - centers[bigger_cluster])**2, axis=1)
        dist[labels == (1 - bigger_cluster)] = dist.min()
        top_k_indices = np.argsort(dist)[-points_to_move:]
        labels[top_k_indices] = 1 - labels[top_k_indices]
        # Re-split data based on k-means labels
        cluster_1 = X[labels == 0]
        cluster_2 = X[labels == 1]

    # Update cluster indices to reflect hierarchy level
    cluster_indices += labels * (2 ** current_level)
        
    # Recur for each cluster
    cluster_indices[labels == 0] = hierarchical_kmeans(
        cluster_1, depth, current_level + 1, cluster_indices[labels == 0]
    )
    cluster_indices[labels == 1] = hierarchical_kmeans(
        cluster_2, depth, current_level + 1, cluster_indices[labels == 1]
    )

    return cluster_indices

if __name__ == "__main__":
    # Example usage
    print('Test 1')
    depth = 4
    i = 7
    binary_tensor = get_binary_path(depth, i)
    print(binary_tensor)  # Outputs: tensor([0, 1, 1, 1])

    print('Test 2')
    # Generate some sample data as PyTorch tensors
    torch.manual_seed(42)
    X = torch.rand(8, 2)  # 100 points in 2D space

    # Perform hierarchical k-means
    depth = 4  # Max depth of the hierarchy
    cluster_indices = hierarchical_kmeans(X, depth)

    # Print cluster indices
    print(cluster_indices)

    print('Test 3')
    print(map_emb_to_path(X, depth))