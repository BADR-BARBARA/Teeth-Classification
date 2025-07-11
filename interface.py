import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- PAGE CONFIGURATION ---
# Sets the page title and icon that appear in the browser tab.
st.set_page_config(
    page_title="Teeth Classification AI",
    page_icon="🦷",
    layout="centered"
)


# --- IMPORTANT: MODEL AND CLASS CONFIGURATION ---
# You MUST update these paths and names for your specific project.

# TODO: 1. Update this path to where your .keras model file is located.
MODEL_PATH = r"C:\Users\HP\Desktop\Teeth-Classification\saved_model\best_baseline_model.keras"

# TODO: 2. Update this list with the names of your 7 classes.
# The order MUST be the same as the order your model was trained on.
# You can find the order in `train_generator.class_indices` from your notebook.
CLASS_NAMES = [
    'CaS','CoS','Gum','MC','OC','OLP','OT'  # Replace with your actual class names
]

# Define the image size your model expects
IMG_HEIGHT = 224
IMG_WIDTH = 224


# --- MODEL LOADING ---
# @st.cache_resource is a decorator that caches the loaded model,
# so it doesn't have to be reloaded every time the user interacts with the app.
@st.cache_resource
def load_keras_model(model_path):
    """
    Loads the Keras model from the specified path.
    Includes error handling for a cleaner user experience.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # Display an error message in the app if the model can't be loaded.
        st.error(f"Error: Could not load the model from {model_path}\n{e}")
        return None

# Load the model using the function
model = load_keras_model(MODEL_PATH)


# --- IMAGE PREPROCESSING ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    """
    # Resize the image to the target dimensions
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert the PIL image to a NumPy array
    img_array = np.array(img)
    
    # If the image is grayscale, convert it to 3 channels (RGB)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
        
    # Remove alpha channel if it exists (for PNGs)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    # Normalize pixel values to the [0, 1] range
    img_array = img_array / 255.0
    
    # Expand dimensions to create a batch of 1
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch


# --- STREAMLIT APP LAYOUT ---

# Main title of the app
st.title("🦷 Dental Image Classifier")

# Description and instructions
st.markdown(
    "Drag and drop a dental image below. The AI will analyze it and predict the tooth type."
)

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a tooth for classification."
)

# Main logic: only proceed if the model is loaded and a file is uploaded
if model is not None and uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("") # Adds a little space

    # Create a button to trigger the classification
    if st.button('Classify Image'):
        # Show a spinner while the model is making a prediction
        with st.spinner('Analyzing the image...'):
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Make a prediction
            prediction = model.predict(processed_image)
            
            # Get the index and confidence of the highest probability
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Get the class name using the index
            predicted_class_name = CLASS_NAMES[predicted_class_index]

        # Display the result with a success message
        st.success(f"**Prediction:** {predicted_class_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")

elif model is None:
    # This message shows if the model failed to load at the start
    st.warning("Please check the model path. The application cannot start without a valid model.")

# --- Optional: Add a footer or about section ---
st.markdown("---")
st.markdown("Developed by an AI Expert for Teeth Classification.")