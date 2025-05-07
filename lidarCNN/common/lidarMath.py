import numpy as np

def vmag(v1):
    """
    Calculates the magnitude of an m x n matrix into an m x 1 vector
    
    Parameters:
    v1 (numpy.ndarray): Input matrix of size m x n
    
    Returns:
    numpy.ndarray: Vector of size m x 1 containing the magnitudes
    """
    # Fetch size of the matrix
    m = v1.shape
    
    # Pre-allocate memory
    v2 = np.zeros((m, 1))
    
    # Calculate magnitude for each row
    for i in range(m):
        v2[i] = np.linalg.norm(v1[i, :])
    
    return v2