"""
Visualization functions for neural network training
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def plot_decision_boundary(network, X, y, title="Decision Boundary"):
    """
    Plot decision boundary with data points
    
    Args:
        network: Trained neural network
        X: Feature data (n_samples, 2)
        y: Labels (n_samples,)
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get decision boundary
    xx, yy, Z = network.get_decision_boundary(resolution=100)
    
    # Plot decision boundary as contourf
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6, vmin=0, vmax=1)
    
    # Add contour lines at 0.5 (decision boundary)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Plot data points
    scatter = ax.scatter(
        X[:, 0], X[:, 1], 
        c=y, 
        cmap='RdBu', 
        s=100, 
        edgecolor='white', 
        linewidth=2,
        vmin=0, 
        vmax=1
    )
    
    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Feature 1 (x)', fontsize=12)
    ax.set_ylabel('Feature 2 (y)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Prediction Probability', fontsize=10)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444', 
                   markersize=10, label='Class 0', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6', 
                   markersize=10, label='Class 1', markeredgecolor='white', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_loss_curve(loss_history, title="Training Loss"):
    """
    Plot loss curve over training
    
    Args:
        loss_history: List of loss values
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if len(loss_history) > 0:
        epochs = range(1, len(loss_history) + 1)
        ax.plot(epochs, loss_history, linewidth=2, color='#ec4899', marker='o', 
                markersize=4, markevery=max(1, len(loss_history) // 20))
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add current loss annotation
        if len(loss_history) > 0:
            last_loss = loss_history[-1]
            ax.annotate(
                f'Current: {last_loss:.4f}',
                xy=(len(loss_history), last_loss),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=10
            )
    else:
        ax.text(0.5, 0.5, 'Start training to see loss curve', 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_network_architecture(network):
    """
    Plot network architecture diagram
    
    Args:
        network: Neural network instance
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Layer positions
    input_layer_x = 0.2
    hidden_layer_x = 0.5
    output_layer_x = 0.8
    
    # Neuron positions
    input_neurons = network.input_size
    hidden_neurons = network.hidden_size
    output_neurons = network.output_size
    
    def get_y_positions(n_neurons, y_center=0.5, spacing=0.1):
        """Calculate y positions for neurons"""
        if n_neurons == 1:
            return [y_center]
        start = y_center - (n_neurons - 1) * spacing / 2
        return [start + i * spacing for i in range(n_neurons)]
    
    input_y = get_y_positions(input_neurons, spacing=0.15)
    hidden_y = get_y_positions(hidden_neurons, spacing=0.1)
    output_y = get_y_positions(output_neurons)
    
    # Draw connections (weights)
    for i, y1 in enumerate(input_y):
        for j, y2 in enumerate(hidden_y):
            weight = network.W1[i, j]
            color = '#3b82f6' if weight > 0 else '#ef4444'
            alpha = min(abs(weight) / 2, 0.8)
            ax.plot([input_layer_x, hidden_layer_x], [y1, y2], 
                   color=color, alpha=alpha, linewidth=abs(weight)*2)
    
    for i, y1 in enumerate(hidden_y):
        for j, y2 in enumerate(output_y):
            weight = network.W2[i, j]
            color = '#3b82f6' if weight > 0 else '#ef4444'
            alpha = min(abs(weight) / 2, 0.8)
            ax.plot([hidden_layer_x, output_layer_x], [y1, y2], 
                   color=color, alpha=alpha, linewidth=abs(weight)*2)
    
    # Draw neurons
    for y in input_y:
        circle = plt.Circle((input_layer_x, y), 0.03, color='#10b981', zorder=10)
        ax.add_patch(circle)
    
    for i, y in enumerate(hidden_y):
        activation = 0 if network.hidden_activations is None else network.hidden_activations[0, i]
        color_intensity = activation
        circle = plt.Circle((hidden_layer_x, y), 0.03, 
                          color=plt.cm.RdYlBu(color_intensity), zorder=10)
        ax.add_patch(circle)
    
    for y in output_y:
        activation = 0 if network.output_activation is None else network.output_activation[0, 0]
        circle = plt.Circle((output_layer_x, y), 0.03, 
                          color=plt.cm.RdYlBu(activation), zorder=10)
        ax.add_patch(circle)
    
    # Labels
    ax.text(input_layer_x, 0.05, 'Input Layer\n(2 neurons)', 
           ha='center', fontsize=10, fontweight='bold')
    ax.text(hidden_layer_x, 0.05, f'Hidden Layer\n({hidden_neurons} neurons)', 
           ha='center', fontsize=10, fontweight='bold')
    ax.text(output_layer_x, 0.05, 'Output Layer\n(1 neuron)', 
           ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Network Architecture', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def display_metrics(network, X, y, epoch, current_loss):
    """
    Display training metrics in Streamlit
    
    Args:
        network: Neural network instance
        X: Feature data
        y: Labels
        epoch: Current epoch
        current_loss: Current loss value
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Epoch", epoch)
    
    with col2:
        st.metric("Loss", f"{current_loss:.4f}" if current_loss is not None else "N/A")
    
    with col3:
        accuracy = network.get_accuracy(X, y)
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col4:
        total_params = (network.input_size * network.hidden_size + 
                       network.hidden_size + 
                       network.hidden_size * network.output_size + 
                       network.output_size)
        st.metric("Parameters", total_params)