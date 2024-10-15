import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
classnames = ['Mild_Demented','Moderate_Demented','Non_Demented','Very_Mild_Demented']

# Load the model
model = tf.keras.models.load_model('./Alzheimer.h5')

def preprocess_image(image):
    # Resize image to 224x224
    image = image.resize((224, 224))

    # Convert to numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values

    # Add channel dimension and convert to RGB
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.repeat(image_array, 3, axis=-1)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# Streamlit app
def main():
    st.title("Alzheimers Prediction")
    st.write("Created by Jwalitha")
    uploaded_file = st.file_uploader("Upload your Brain MRI ")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img,caption='Uploaded Image.',use_column_width=True)
        image_array = preprocess_image(img)
        prediction = model.predict(image_array)
        ind = np.argmax(prediction[0])
        st.write("Predicted Class:", classnames[ind])

if __name__ == '__main__':
    main()
