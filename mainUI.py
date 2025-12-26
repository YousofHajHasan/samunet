import streamlit as st
import numpy as np
from PIL import Image

from main import process_image, load_yolo_model

st.set_page_config(page_title="SamuNet Spine Analyzer", layout="wide")

st.title("SamuNet Spine Analyzer")
st.markdown("Analyze vertebrae and calculate spondylolisthesis slip percentages")

# Helper function to resize image maintaining aspect ratio
def resize_to_height(img, target_height=780):
    """Resize image to target height while maintaining aspect ratio"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    w, h = img.size
    ratio = target_height / h
    new_width = int(w * ratio)
    return img.resize((new_width, target_height), Image.Resampling.LANCZOS)

# Initialize session state for results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Sidebar controls
st.sidebar.header("Visualization Options")
visualize_lines_toggle = st.sidebar.toggle("Visualize Lines", value=True)
visualize_corners_toggle = st.sidebar.toggle("Visualize Corners", value=True)

# Cache the YOLO model to avoid reloading
@st.cache_resource
def get_yolo_model():
    return load_yolo_model()

# File uploader
uploaded_file = st.file_uploader("Choose a spine X-ray image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # Resize for display
    image_display = resize_to_height(image, target_height=780)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_display, use_container_width=False)
    
    # Process button
    if st.button("Analyze Spine", type="primary"):
        with st.spinner("Processing image... This may take a moment."):
            try:
                # Load YOLO model (cached)
                yolo_model = get_yolo_model()
                
                # Process the image (generates all visualization combinations)
                st.session_state.analysis_results = process_image(
                    image_rgb=image_np,
                    yolo_model=yolo_model
                )
                
                st.success("Analysis complete!")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        
        # Get the appropriate image based on toggle states
        current_image = results['images'][(visualize_lines_toggle, visualize_corners_toggle)]
        result_display = resize_to_height(current_image, target_height=780)
        
        with col2:
            st.subheader("Analysis Result")
            st.image(result_display, use_container_width=False)
        
        # Display slip percentages
        st.sidebar.markdown("---")
        st.sidebar.subheader("Slip Percentages")
        
        for (v1, v2), percentage in results['slip_percentages'].items():
            st.sidebar.metric(
                label=f"{v1.upper()} - {v2.upper()}",
                value=f"{percentage:.2f}%"
            )
        
        # Display vertebrae detected
        st.sidebar.markdown("---")
        st.sidebar.subheader("Vertebrae Detected")
        detected = list(results['vertebra_data'].keys())
        st.sidebar.write(", ".join([v.upper() for v in detected]))

else:
    # Reset results when no file is uploaded
    st.session_state.analysis_results = None
    st.info("Please upload a spine X-ray image to get started.")

# Footer
st.markdown("---")
st.markdown("*SamuNet Spine Analysis Application - Spondylolisthesis Detection*")
