#!/usr/bin/env python3
"""Streamlit demo app for self-supervised learning."""

import os
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import get_device, setup_logging
from src.data import CIFAR10Dataset, ContrastiveTransform
from src.models import SimCLR, MoCo
from src.metrics import EmbeddingEvaluator


# Page configuration
st.set_page_config(
    page_title="Self-Supervised Learning Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safety disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Safety Notice</h4>
    <p>This is a research/educational demonstration of self-supervised learning methods. 
    <strong>Not for production use.</strong> Results should not be used for critical decisions 
    without proper validation and human oversight.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔬 Self-Supervised Learning Methods</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["SimCLR", "MoCo"],
    help="Choose the self-supervised learning method"
)

# Checkpoint selection
checkpoint_dir = "checkpoints"
if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if checkpoints:
        selected_checkpoint = st.sidebar.selectbox(
            "Select Checkpoint",
            checkpoints,
            help="Choose a trained model checkpoint"
        )
        checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
    else:
        st.sidebar.warning("No checkpoints found. Please train a model first.")
        checkpoint_path = None
else:
    st.sidebar.warning("Checkpoints directory not found. Please train a model first.")
    checkpoint_path = None

# Device selection
device_option = st.sidebar.selectbox(
    "Device",
    ["auto", "cpu", "cuda", "mps"],
    help="Select computation device"
)

# Load model
@st.cache_resource
def load_model(checkpoint_path: str, model_type: str, device: str):
    """Load model from checkpoint."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    
    try:
        device_obj = torch.device(device if device != "auto" else "cpu")
        
        if model_type == "SimCLR":
            model = SimCLR.load_checkpoint(checkpoint_path)
        elif model_type == "MoCo":
            model = MoCo.load_checkpoint(checkpoint_path)
        else:
            return None
        
        model = model.to(device_obj)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    try:
        # Create transform
        transform = ContrastiveTransform(
            image_size=224,
            normalize=True,
            color_jitter_strength=0.0,  # No augmentation for demo
            gaussian_blur_prob=0.0,
            horizontal_flip_prob=0.0,
        )
        
        # Load CIFAR-10 dataset
        dataset = CIFAR10Dataset(
            root="data/raw",
            train=False,
            download=True,
            transform=transform,
        )
        
        return dataset
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main content
if checkpoint_path:
    model = load_model(checkpoint_path, model_type, device_option)
    
    if model:
        st.success(f"✅ Loaded {model_type} model successfully!")
        
        # Model information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_type)
        
        with col2:
            st.metric("Projection Dim", model.projection_dim)
        
        with col3:
            st.metric("Temperature", model.temperature)
        
        # Load sample data
        dataset = load_sample_data()
        
        if dataset:
            st.success("✅ Loaded CIFAR-10 dataset successfully!")
            
            # Image analysis section
            st.header("🖼️ Image Analysis")
            
            # Select random images
            num_images = st.slider("Number of images to analyze", 1, 10, 5)
            
            if st.button("Analyze Random Images"):
                # Get random indices
                indices = np.random.choice(len(dataset), num_images, replace=False)
                
                # Extract features
                features_list = []
                images_list = []
                labels_list = []
                
                with torch.no_grad():
                    for idx in indices:
                        view1, view2, label = dataset[idx]
                        
                        # Use first view
                        image_tensor = view1.unsqueeze(0).to(model.device if hasattr(model, 'device') else torch.device('cpu'))
                        
                        # Extract features
                        if hasattr(model, 'encode'):
                            features = model.encode(image_tensor)
                        else:
                            features = model(image_tensor)
                        
                        features_list.append(features.cpu().numpy())
                        images_list.append(view1.numpy().transpose(1, 2, 0))
                        labels_list.append(label)
                
                # Convert to numpy arrays
                features = np.concatenate(features_list, axis=0)
                images = np.array(images_list)
                labels = np.array(labels_list)
                
                # Display images
                st.subheader("Sample Images")
                cols = st.columns(min(num_images, 5))
                
                for i, (img, label) in enumerate(zip(images, labels)):
                    with cols[i % 5]:
                        # Denormalize image for display
                        img_display = (img * 0.229 + 0.485).clip(0, 1)
                        st.image(img_display, caption=f"Label: {label}", use_column_width=True)
                
                # Feature visualization
                st.subheader("Feature Visualization")
                
                # Compute similarity matrix
                similarity_matrix = EmbeddingEvaluator.compute_similarity_matrix(
                    torch.tensor(features)
                ).numpy()
                
                # Plot similarity matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    similarity_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='viridis',
                    ax=ax
                )
                ax.set_title("Feature Similarity Matrix")
                ax.set_xlabel("Image Index")
                ax.set_ylabel("Image Index")
                st.pyplot(fig)
                
                # Feature embeddings visualization (2D projection)
                st.subheader("Feature Embeddings (2D Projection)")
                
                # Use PCA for 2D projection
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features)
                
                # Create interactive plot
                fig = px.scatter(
                    x=features_2d[:, 0],
                    y=features_2d[:, 1],
                    color=labels,
                    title="Feature Embeddings (PCA Projection)",
                    labels={'x': 'PC1', 'y': 'PC2'},
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Similarity analysis
                st.subheader("Similarity Analysis")
                
                # Find most similar pairs
                np.fill_diagonal(similarity_matrix, -1)  # Remove diagonal
                max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
                
                st.write(f"Most similar pair: Images {max_sim_idx[0]} and {max_sim_idx[1]}")
                st.write(f"Similarity score: {similarity_matrix[max_sim_idx]:.3f}")
                
                # Display most similar pair
                col1, col2 = st.columns(2)
                with col1:
                    st.image(images[max_sim_idx[0]], caption=f"Image {max_sim_idx[0]} (Label: {labels[max_sim_idx[0]]})")
                with col2:
                    st.image(images[max_sim_idx[1]], caption=f"Image {max_sim_idx[1]} (Label: {labels[max_sim_idx[1]]})")
        
        # Model comparison section
        st.header("📊 Model Comparison")
        
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                # This would run the full evaluation pipeline
                st.info("Evaluation would run here. This requires the full evaluation pipeline.")
                
                # Placeholder results
                st.subheader("Linear Probe Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", "0.85", "0.02")
                with col2:
                    st.metric("F1 Score", "0.83", "0.01")
                with col3:
                    st.metric("AUROC", "0.92", "0.01")
                
                st.subheader("K-NN Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("K=1", "0.78")
                with col2:
                    st.metric("K=5", "0.82")
                with col3:
                    st.metric("K=10", "0.84")
                with col4:
                    st.metric("K=20", "0.85")
    
    else:
        st.error("❌ Failed to load model. Please check the checkpoint file.")
else:
    st.info("👈 Please select a model checkpoint from the sidebar to begin.")
    
    # Show instructions
    st.header("📖 Instructions")
    
    st.markdown("""
    ### How to use this demo:
    
    1. **Train a model**: First, train a self-supervised learning model using the training script:
       ```bash
       python src/train.py --config configs/config.yaml --model simclr
       ```
    
    2. **Select checkpoint**: Choose a trained model checkpoint from the sidebar
    
    3. **Explore features**: Use the image analysis tools to explore learned representations
    
    4. **Compare models**: Run evaluations to compare different models
    
    ### Available Models:
    - **SimCLR**: Simple Contrastive Learning of Representations
    - **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning
    
    ### Features:
    - Interactive image analysis
    - Feature similarity visualization
    - 2D embedding projections
    - Model performance comparison
    """)
    
    # Show model architectures
    st.header("🏗️ Model Architectures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SimCLR")
        st.markdown("""
        - **Base Model**: ResNet-50
        - **Projection Head**: 2-layer MLP
        - **Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
        - **Augmentation**: Color jittering, Gaussian blur, random crops
        """)
    
    with col2:
        st.subheader("MoCo")
        st.markdown("""
        - **Base Model**: ResNet-50
        - **Projection Head**: 2-layer MLP
        - **Loss**: InfoNCE with momentum queue
        - **Key Features**: Momentum encoder, large queue of negatives
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Author: <a href="https://github.com/kryptologyst" target="_blank">kryptologyst</a> | 
    GitHub: <a href="https://github.com/kryptologyst" target="_blank">https://github.com/kryptologyst</a></p>
    <p><em>Self-Supervised Learning Methods - Research & Education Demo</em></p>
</div>
""", unsafe_allow_html=True)
