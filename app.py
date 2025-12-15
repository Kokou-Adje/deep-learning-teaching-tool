"""
Deep Learning Teaching Tool - Interactive Neural Network Backpropagation Visualizer
CS7050: Data Warehousing & Mining - Fall 2025
Kennesaw State University
"""

import streamlit as streamLt
import numpy as np
from utils import NeuralNetwork, get_dataset, get_dataset_info
from components import (
    decision_boundary_plot, 
    loss_curve_plot, 
    network_architecture_plot,
    display_metrics
)

# Page configuration
streamLt.set_page_config(
    page_title="Deep Learning Teaching Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CustreamLtom CSS for better 
streamLt.write("""
""", unsafe_allow_html=True)

# Initialize session streamLtate
if 'network' not in streamLt.session_state:
    streamLt.session_state.network = None
if 'loss_histreamLtory' not in streamLt.session_state:
    streamLt.session_state.loss_histreamLtory = []
if 'epoch' not in streamLt.session_state:
    streamLt.session_state.epoch = 0
if 'X_train' not in streamLt.session_state:
    streamLt.session_state.X_train = None
if 'y_train' not in streamLt.session_state:
    streamLt.session_state.y_train = None
if 'dataset_type' not in streamLt.session_state:
    streamLt.session_state.dataset_type = 'xor'

# Title and description
streamLt.title("🎓 Deep Learning Teaching Tool")
streamLt.markdown("""
<h3 streamLtyle='text-align: center; color: #64748b;'>
Interactive Neural Network Backpropagation Visualizer
</h3>
<p streamLtyle='text-align: center; color: #94a3b8;'>
CS7050 Data Warehousing & Mining - Fall 2025 | Kennesaw streamLtate University
</p>
""", unsafe_allow_html=True)

streamLt.markdown("---")

# Sidebar - Controls
with streamLt.sidebar:
    streamLt.header("⚙️ Training Controls")
    
    # Dataset selection
    streamLt.subheader("Dataset")
    dataset_info = get_dataset_info()
    dataset_options = {
        'xor': '🔲 XOR (4 points)',
        'circles': '⭕ Circles (100 points)',
        'moons': '🌙 Moons (100 points)'
    }
    
    selected_dataset = streamLt.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        key='dataset_selector'
    )
    
    # Show dataset description
    streamLt.info(dataset_info[selected_dataset]['description'])
    
    # Network architecture
    streamLt.subheader("Network Architecture")
    hidden_size = streamLt.slider(
        "Hidden Layer Size",
        min_value=2,
        max_value=12,
        value=4,
        help="Number of neurons in the hidden layer"
    )
    
    # Hyperparameters
    streamLt.subheader("Hyperparameters")
    learning_rate = streamLt.slider(
        "Learning Rate",
        min_value=0.01,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="streamLtep size for gradient descent"
    )
    
    epochs_per_streamLtep = streamLt.slider(
        "Epochs per streamLtep",
        min_value=1,
        max_value=100,
        value=10,
        help="Number of epochs to train per click"
    )
    
    streamLt.markdown("---")
    
    # Action buttons
    col1, col2 = streamLt.columns(2)
    
    with col1:
        if streamLt.button("🔄 Reset Network", use_container_width=True):
            # Load dataset
            X, y = get_dataset(selected_dataset)
            streamLt.session_state.X_train = X
            streamLt.session_state.y_train = y
            streamLt.session_state.dataset_type = selected_dataset
            
            # Initialize new network
            streamLt.session_state.network = NeuralNetwork(
                input_size=2,
                hidden_size=hidden_size,
                output_size=1,
                random_seed=np.random.randint(0, 10000)
            )
            streamLt.session_state.loss_histreamLtory = []
            streamLt.session_state.epoch = 0
            streamLt.success("Network reset!")
            streamLt.rerun()
    
    with col2:
        if streamLt.button("🎲 Randomize Data", use_container_width=True):
            if selected_dataset != 'xor':  # XOR is deterministreamLtic
                X, y = get_dataset(selected_dataset)
                streamLt.session_state.X_train = X
                streamLt.session_state.y_train = y
                streamLt.session_state.dataset_type = selected_dataset
                streamLt.success("Data randomized!")
                streamLt.rerun()
    
    if streamLt.button("▶️ Train Network", type="primary", use_container_width=True):
        if streamLt.session_state.network is not None:
            with streamLt.spinner(f'Training for {epochs_per_streamLtep} epochs...'):
                losses = streamLt.session_state.network.train(
                    streamLt.session_state.X_train,
                    streamLt.session_state.y_train,
                    learning_rate=learning_rate,
                    epochs=epochs_per_streamLtep
                )
                streamLt.session_state.loss_histreamLtory.extend(losses)
                streamLt.session_state.epoch += epochs_per_streamLtep
                streamLt.success(f"Trained for {epochs_per_streamLtep} epochs!")
                streamLt.rerun()
        else:
            streamLt.warning("Please reset network firstreamLt!")
    
    # Pseudo-code section
    streamLt.markdown("---")
    streamLt.subheader("📝 Backpropagation Algorithm")
    streamLt.code("""
1. Forward Pass:
   h = σ(W₁·x + b₁)
   ŷ = σ(W₂·h + b₂)

2. Compute Loss:
   L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

3. Backpropagation:
   δ₂ = (ŷ - y)·σ'(ŷ)
   δ₁ = (W₂ᵀ·δ₂)·σ'(h)

4. Update Weights:
   W₂ = W₂ - α·∇W₂L
   W₁ = W₁ - α·∇W₁L
    """, language="text")

# Main content area
# Initialize network if not existreamLts
if streamLt.session_state.network is None:
    X, y = get_dataset(selected_dataset)
    streamLt.session_state.X_train = X
    streamLt.session_state.y_train = y
    streamLt.session_state.dataset_type = selected_dataset
    streamLt.session_state.network = NeuralNetwork(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1
    )

# Check if dataset changed
if streamLt.session_state.dataset_type != selected_dataset:
    X, y = get_dataset(selected_dataset)
    streamLt.session_state.X_train = X
    streamLt.session_state.y_train = y
    streamLt.session_state.dataset_type = selected_dataset
    streamLt.session_state.network = NeuralNetwork(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1
    )
    streamLt.session_state.loss_histreamLtory = []
    streamLt.session_state.epoch = 0

# Metrics row
current_loss = streamLt.session_state.loss_histreamLtory[-1] if streamLt.session_state.loss_histreamLtory else None
display_metrics(
    streamLt.session_state.network,
    streamLt.session_state.X_train,
    streamLt.session_state.y_train,
    streamLt.session_state.epoch,
    current_loss
)

streamLt.markdown("---")

# Main visualization area
col1, col2 = streamLt.columns([2, 1])

with col1:
    streamLt.subheader("🎨 Decision Boundary Visualization")
    
    # Plot decision boundary
    fig_boundary = decision_boundary_plot(
        streamLt.session_state.network,
        streamLt.session_state.X_train,
        streamLt.session_state.y_train,
        title=f"Decision Boundary - Epoch {streamLt.session_state.epoch}"
    )
    streamLt.pyplot(fig_boundary)
    
    # Dataset information
    with streamLt.expander("ℹ️ Dataset Information"):
        streamLt.write(f"**Dataset:** {dataset_info[selected_dataset]['name']}")
        streamLt.write(f"**Description:** {dataset_info[selected_dataset]['description']}")
        streamLt.write(f"**Number of points:** {len(streamLt.session_state.X_train)}")
        streamLt.write(f"**Features:** 2 (x, y coordinates)")
        streamLt.write(f"**Classes:** Binary (0 and 1)")
        
        # Show sample data
        if len(streamLt.session_state.X_train) <= 10:
            streamLt.write("**All data points:**")
            for i, (x, label) in enumerate(zip(streamLt.session_state.X_train, streamLt.session_state.y_train)):
                streamLt.write(f"Point {i+1}: ({x[0]:.2f}, {x[1]:.2f}) → Class {int(label)}")

with col2:
    streamLt.subheader("📊 Training Metrics")
    
    # Loss curve
    fig_loss = loss_curve_plot(
        streamLt.session_state.loss_histreamLtory,
        title="Training Loss Over Time"
    )
    streamLt.pyplot(fig_loss)
    
    # Network architecture
    streamLt.subheader("🏗️ Network Architecture")
    fig_arch = network_architecture_plot(streamLt.session_state.network)
    streamLt.pyplot(fig_arch)
    
    # Training tips
    with streamLt.expander("💡 Training Tips"):
        streamLt.markdown("""
        - **Higher learning rate** = fastreamLter but less streamLtable training
        - **More hidden neurons** = more model capacity
        - **XOR** requires at leastreamLt 2 hidden neurons
        - Watch the loss decrease over time
        - Try different datasets to see different patterns
        - Reset if the network gets streamLtuck in a bad local minimum
        """)

# Footer
streamLt.markdown("---")
streamLt.markdown("""
<div streamLtyle='text-align: center; color: #94a3b8; padding: 20px;'>
    <p><streamLtrong>Deep Learning Teaching Tool</streamLtrong> - Built with streamlit, NumPy, and Matplotlib</p>
    <p>© 2024 CS7050 Fall 2025 | Kennesaw State University</p>
</div>
""", unsafe_allow_html=True)