# utils/__init__.py
"""Utility modules for neural network and datasets"""

from .neural_network import NeuralNetwork
from .datasets import (
    generate_xor, 
    generate_circles, 
    generate_moons, 
    get_dataset,
    get_dataset_info
)

__all__ = [
    'NeuralNetwork',
    'generate_xor',
    'generate_circles',
    'generate_moons',
    'get_dataset',
    'get_dataset_info'
]
