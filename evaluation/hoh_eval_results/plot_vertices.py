import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

def plot_vertices_3d(vertices):
    """
    Plots 3D scatter plot of vertices.
    
    Parameters:
        vertices (list of tuples): A list of tuples where each tuple contains (x, y, z) coordinates of a vertex.
    """
    # Convert vertices to numpy array
    vertices = np.array(vertices)
    
    # Remove invalid values (NaN or inf)
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    
    # Plot the vertices
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_kde(vertices):
    """Plots KDE for 3D vertices."""
    # Convert vertices to numpy array and transpose to match (dim, num_samples) format
    vertices = np.array(vertices).T
    
    kde = gaussian_kde(vertices)
    
    x = np.linspace(vertices[0].min(), vertices[0].max(), 50)
    y = np.linspace(vertices[1].min(), vertices[1].max(), 50)
    z = np.linspace(vertices[2].min(), vertices[2].max(), 50)
    
    X, Y, Z = np.meshgrid(x, y, z)
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    
    density = kde(positions)
    density = density.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X[:, :, 0], Y[:, :, 0], density[:, :, 0], cmap='viridis')
    ax.scatter(vertices[0], vertices[1], vertices[2], c='r', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    plt.show()

