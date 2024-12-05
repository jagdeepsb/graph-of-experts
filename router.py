import torch
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

def get_binary_path(depth, i):
    '''
    Based on the embedding vector that the input image maps to, output
    a corresponding path in the graph
    '''
    # Convert `i` to binary and pad to `depth` bits
    binary_representation = bin(i)[2:].zfill(depth)
    
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
    return {i: get_binary_path(depth, cluster_indices[i]) for i in range(len(embs))}

def hierarchical_kmeans(X, depth, current_level=0, cluster_indices=None):
    """
    Hierarchical K-Means Clustering for k=2, returning cluster indices.
    
    X: torch tensor of shape [n, d] (n points, each of dimension d)
    depth: int, maximum depth of the hierarchy
    current_level: int, the current depth in the hierarchy
    cluster_indices: torch tensor of shape [n], stores cluster indices of points
    
    Returns:
        cluster_indices: torch tensor of shape [n], the cluster index of each point
    """
    if cluster_indices is None:
        # Initialize cluster indices as 0 for all points
        cluster_indices = torch.zeros(X.shape[0], dtype=torch.int32)

    if current_level == depth or X.shape[0] <= 1:
        # Return the current cluster indices if depth is reached or only one point
        return cluster_indices

    # Convert PyTorch tensor to NumPy array for sklearn KMeans compatibility
    X_np = X.numpy()

    # Apply K-Means with k=2
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_np)

    # Assign points to clusters
    labels = torch.tensor(kmeans.labels_, dtype=torch.int32)

    # Update cluster indices to reflect hierarchy level
    cluster_indices += labels * (2 ** current_level)

    # Split data based on k-means labels
    cluster_1 = X[labels == 0]
    cluster_2 = X[labels == 1]

    print(f'Split{current_level} into {len(cluster_1)} and {len(cluster_2)}')

    # Recur for each cluster
    cluster_indices[labels == 0] = hierarchical_kmeans(
        cluster_1, depth, current_level + 1, cluster_indices[labels == 0]
    )
    cluster_indices[labels == 1] = hierarchical_kmeans(
        cluster_2, depth, current_level + 1, cluster_indices[labels == 1]
    )

    return cluster_indices

def hierarchical_spectral_clustering(X, depth, current_level=0, cluster_indices=None):
    """
    Hierarchical Spectral Clustering for k=2, returning cluster indices.
    
    X: torch tensor of shape [n, d] (n points, each of dimension d)
    depth: int, maximum depth of the hierarchy
    current_level: int, the current depth in the hierarchy
    cluster_indices: torch tensor of shape [n], stores cluster indices of points
    
    Returns:
        cluster_indices: torch tensor of shape [n], the cluster index of each point
    """
    if cluster_indices is None:
        # Initialize cluster indices as 0 for all points
        cluster_indices = torch.zeros(X.shape[0], dtype=torch.int32)

    if current_level == depth or X.shape[0] <= 1:
        # Return the current cluster indices if depth is reached or only one point
        return cluster_indices

    # Convert PyTorch tensor to NumPy array for sklearn SpectralClustering compatibility
    X_np = X.numpy()

    # Apply Spectral Clustering with k=2
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42, n_neighbors=X.shape[0]-1, assign_labels='discretize')
    labels = spectral.fit_predict(X_np)

    # Convert labels to a torch tensor
    labels = torch.tensor(labels, dtype=torch.int32)

    # Update cluster indices to reflect hierarchy level
    cluster_indices += labels * (2 ** current_level)

    # Split data based on spectral clustering labels
    cluster_1 = X[labels == 0]
    cluster_2 = X[labels == 1]

    print(f'Split{current_level} into {len(cluster_1)} and {len(cluster_2)}')

    # Recur for each cluster
    cluster_indices[labels == 0] = hierarchical_spectral_clustering(
        cluster_1, depth, current_level + 1, cluster_indices[labels == 0]
    )
    cluster_indices[labels == 1] = hierarchical_spectral_clustering(
        cluster_2, depth, current_level + 1, cluster_indices[labels == 1]
    )

    return cluster_indices

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
depth = 3  # Max depth of the hierarchy
cluster_indices = hierarchical_kmeans(X, depth)

# Print cluster indices
print(cluster_indices)

print('Test 3')
cluster_indices = hierarchical_spectral_clustering(X, depth)

# Print cluster indices
print(cluster_indices)

print('Test 4')
print(map_emb_to_path(X, depth))