"""
Dataset generation functions for 2D binary classification
"""

import numpy as np


def generate_xor():
    """
    Generate XOR dataset (4 points)
    Classic non-linearly separable problem
    """
    X = np.array([
        [0.2, 0.2],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.8, 0.8]
    ])
    y = np.array([0, 1, 1, 0])
    return X, y


def generate_circles(n_samples=100, noise=0.05):
    """
    Generate concentric circles dataset
    
    Args:
        n_samples: Total number of points (split evenly between classes)
        noise: Standard deviation of Gaussian noise
    
    Returns:
        X: (n_samples, 2) array of coordinates
        y: (n_samples,) array of labels
    """
    n_per_class = n_samples // 2
    
    # Inner circle (class 0)
    angles_inner = np.random.uniform(0, 2 * np.pi, n_per_class)
    radius_inner = 0.25 + np.random.randn(n_per_class) * noise
    X_inner = np.column_stack([
        0.5 + radius_inner * np.cos(angles_inner),
        0.5 + radius_inner * np.sin(angles_inner)
    ])
    
    # Outer circle (class 1)
    angles_outer = np.random.uniform(0, 2 * np.pi, n_per_class)
    radius_outer = 0.65 + np.random.randn(n_per_class) * noise
    X_outer = np.column_stack([
        0.5 + radius_outer * np.cos(angles_outer),
        0.5 + radius_outer * np.sin(angles_outer)
    ])
    
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X, y


def generate_moons(n_samples=100, noise=0.1):
    """
    Generate two interleaving moons dataset
    
    Args:
        n_samples: Total number of points
        noise: Standard deviation of Gaussian noise
    
    Returns:
        X: (n_samples, 2) array of coordinates
        y: (n_samples,) array of labels
    """
    n_per_class = n_samples // 2
    
    # Upper moon (class 0)
    angles_upper = np.linspace(0, np.pi, n_per_class)
    X_upper = np.column_stack([
        0.5 + 0.4 * np.cos(angles_upper) + np.random.randn(n_per_class) * noise,
        0.3 + 0.4 * np.sin(angles_upper) + np.random.randn(n_per_class) * noise
    ])
    
    # Lower moon (class 1)
    angles_lower = np.linspace(0, np.pi, n_per_class)
    X_lower = np.column_stack([
        0.5 - 0.4 * np.cos(angles_lower) + np.random.randn(n_per_class) * noise,
        0.7 - 0.4 * np.sin(angles_lower) + np.random.randn(n_per_class) * noise
    ])
    
    X = np.vstack([X_upper, X_lower])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X, y


def get_dataset_info():
    """Return information about available datasets"""
    return {
        'xor': {
            'name': 'XOR',
            'description': 'Classic non-linearly separable problem (4 points)',
            'points': 4
        },
        'circles': {
            'name': 'Concentric Circles',
            'description': 'Two concentric circles (100 points)',
            'points': 100
        },
        'moons': {
            'name': 'Two Moons',
            'description': 'Interleaving crescents (100 points)',
            'points': 100
        }
    }


def get_dataset(dataset_name, n_samples=100):
    """
    Get dataset by name
    
    Args:
        dataset_name: 'xor', 'circles', or 'moons'
        n_samples: Number of samples (ignored for XOR)
    
    Returns:
        X: Feature array
        y: Label array
    """
    if dataset_name == 'xor':
        return generate_xor()
    elif dataset_name == 'circles':
        return generate_circles(n_samples)
    elif dataset_name == 'moons':
        return generate_moons(n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")