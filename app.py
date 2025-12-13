"""
Deep Learning Teaching Tool - Interactive Neural Network Backpropagation Visualizer
CS7050: Data Warehousing & Mining - Fall 2025
Kennesaw State University
"""

import streamlit as st
import numpy as np
from utils import NeuralNetwork, get_dataset, get_dataset_info
from components import (
    plot_decision_boundary, 
    plot_loss_curve, 
    plot_network_architecture,
    display_metrics
)

# Page configuration
st.set_page_config(
    page_title="Deep Learning Teaching Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .markdown-text-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1 {
        color: #1e293b;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'network' not in st.session_state:
    st.session_state.network = None
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = 'xor'

# Title and description
st.title("🎓 Deep Learning Teaching Tool")
st.markdown("""
<h3 style='text-align: center; color: #64748b;'>
Interactive Neural Network Backpropagation Visualizer
</h3>
<p style='text-align: center; color: #94a3b8;'>
CS7050 Data Warehousing & Mining - Fall 2025 | Kennesaw State University
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Controls
with st.sidebar:
    st.header("⚙️ Training Controls")
    
    # Dataset selection
    st.subheader("Dataset")
    dataset_info = get_dataset_info()
    dataset_options = {
        'xor': '🔲 XOR (4 points)',
        'circles': '⭕ Circles (100 points)',
        'moons': '🌙 Moons (100 points)'
    }
    
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        key='dataset_selector'
    )
    
    # Show dataset description
    st.info(dataset_info[selected_dataset]['description'])
    
    # Network architecture
    st.subheader("Network Architecture")
    hidden_size = st.slider(
        "Hidden Layer Size",
        min_value=2,
        max_value=12,
        value=4,
        help="Number of neurons in the hidden layer"
    )
    
    # Hyperparameters
    st.subheader("Hyperparameters")
    learning_rate = st.slider(
        "Learning Rate",
        min_value=0.01,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Step size for gradient descent"
    )
    
    epochs_per_step = st.slider(
        "Epochs per Step",
        min_value=1,
        max_value=100,
        value=10,
        help="Number of epochs to train per click"
    )
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset Network", use_container_width=True):
            # Load dataset
            X, y = get_dataset(selected_dataset)
            st.session_state.X_train = X
            st.session_state.y_train = y
            st.session_state.dataset_type = selected_dataset
            
            # Initialize new network
            st.session_state.network = NeuralNetwork(
                input_size=2,
                hidden_size=hidden_size,
                output_size=1,
                random_seed=np.random.randint(0, 10000)
            )
            st.session_state.loss_history = []
            st.session_state.epoch = 0
            st.success("Network reset!")
            st.rerun()
    
    with col2:
        if st.button("🎲 Randomize Data", use_container_width=True):
            if selected_dataset != 'xor':  # XOR is deterministic
                X, y = get_dataset(selected_dataset)
                st.session_state.X_train = X
                st.session_state.y_train = y
                st.session_state.dataset_type = selected_dataset
                st.success("Data randomized!")
                st.rerun()
    
    if st.button("▶️ Train Network", type="primary", use_container_width=True):
        if st.session_state.network is not None:
            with st.spinner(f'Training for {epochs_per_step} epochs...'):
                losses = st.session_state.network.train(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    learning_rate=learning_rate,
                    epochs=epochs_per_step
                )
                st.session_state.loss_history.extend(losses)
                st.session_state.epoch += epochs_per_step
                st.success(f"Trained for {epochs_per_step} epochs!")
                st.rerun()
        else:
            st.warning("Please reset network first!")
    
    # Pseudo-code section
    st.markdown("---")
    st.subheader("📝 Backpropagation Algorithm")
    st.code("""
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
# Initialize network if not exists
if st.session_state.network is None:
    X, y = get_dataset(selected_dataset)
    st.session_state.X_train = X
    st.session_state.y_train = y
    st.session_state.dataset_type = selected_dataset
    st.session_state.network = NeuralNetwork(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1
    )

# Check if dataset changed
if st.session_state.dataset_type != selected_dataset:
    X, y = get_dataset(selected_dataset)
    st.session_state.X_train = X
    st.session_state.y_train = y
    st.session_state.dataset_type = selected_dataset
    st.session_state.network = NeuralNetwork(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1
    )
    st.session_state.loss_history = []
    st.session_state.epoch = 0

# Metrics row
current_loss = st.session_state.loss_history[-1] if st.session_state.loss_history else None
display_metrics(
    st.session_state.network,
    st.session_state.X_train,
    st.session_state.y_train,
    st.session_state.epoch,
    current_loss
)

st.markdown("---")

# Main visualization area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎨 Decision Boundary Visualization")
    
    # Plot decision boundary
    fig_boundary = plot_decision_boundary(
        st.session_state.network,
        st.session_state.X_train,
        st.session_state.y_train,
        title=f"Decision Boundary - Epoch {st.session_state.epoch}"
    )
    st.pyplot(fig_boundary)
    
    # Dataset information
    with st.expander("ℹ️ Dataset Information"):
        st.write(f"**Dataset:** {dataset_info[selected_dataset]['name']}")
        st.write(f"**Description:** {dataset_info[selected_dataset]['description']}")
        st.write(f"**Number of points:** {len(st.session_state.X_train)}")
        st.write(f"**Features:** 2 (x, y coordinates)")
        st.write(f"**Classes:** Binary (0 and 1)")
        
        # Show sample data
        if len(st.session_state.X_train) <= 10:
            st.write("**All data points:**")
            for i, (x, label) in enumerate(zip(st.session_state.X_train, st.session_state.y_train)):
                st.write(f"Point {i+1}: ({x[0]:.2f}, {x[1]:.2f}) → Class {int(label)}")

with col2:
    st.subheader("📊 Training Metrics")
    
    # Loss curve
    fig_loss = plot_loss_curve(
        st.session_state.loss_history,
        title="Training Loss Over Time"
    )
    st.pyplot(fig_loss)
    
    # Network architecture
    st.subheader("🏗️ Network Architecture")
    fig_arch = plot_network_architecture(st.session_state.network)
    st.pyplot(fig_arch)
    
    # Training tips
    with st.expander("💡 Training Tips"):
        st.markdown("""
        - **Higher learning rate** = faster but less stable training
        - **More hidden neurons** = more model capacity
        - **XOR** requires at least 2 hidden neurons
        - Watch the loss decrease over time
        - Try different datasets to see different patterns
        - Reset if the network gets stuck in a bad local minimum
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 20px;'>
    <p><strong>Deep Learning Teaching Tool</strong> - Built with Streamlit, NumPy, and Matplotlib</p>
    <p>© 2024 CS7050 Fall 2025 | Kennesaw State University</p>
</div>
""", unsafe_allow_html=True)