import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to display images
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

# Function to load and preprocess images
def load_img(image_file):
    img = tf.image.decode_jpeg(image_file, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, (256, 256))  # Resize image to fit model input
    img = img[tf.newaxis, :]
    return img

# Function to handle image upload
def handle_image_upload(key):
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=key)
    
    if uploaded_file is not None:
        # Read image as bytes
        image_bytes = uploaded_file.read()

        # Load and preprocess the image
        img = tf.image.decode_jpeg(image_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        # Check if the image size is 256x256, otherwise resize
        if img.shape[0] != 256 or img.shape[1] != 256:
            img = tf.image.resize(img, (256, 256))

        img = img[tf.newaxis, :]  # Add batch dimension
        return img
    return None

# Load the pre-trained style transfer model from TensorFlow Hub
@st.cache_resource
def load_hub_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Streamlit UI
st.title("Style Transfer")
st.write("Upload a content image and a style image to apply the style transfer.")

# Upload content and style images with unique keys
content_image = handle_image_upload(key="content_image_upload")
style_image = handle_image_upload(key="style_image_upload")

if content_image is not None and style_image is not None:
    st.image(tensor_to_image(content_image[0]), caption="Content Image", use_column_width=True)
    st.image(tensor_to_image(style_image[0]), caption="Style Image", use_column_width=True)

    # Perform style transfer when both images are uploaded
    if st.button("Apply Style Transfer"):
        # Load the style transfer model
        hub_model = load_hub_model()

        # Perform style transfer
        stylized_image = hub_model(tf.constant(content_image, dtype=tf.float32), tf.constant(style_image, dtype=tf.float32))[0]

        # Convert tensor to image
        output_image = tensor_to_image(stylized_image)

        # Display the output image
        st.image(output_image, caption="Stylized Image", use_column_width=True)
