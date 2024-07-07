import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Config function
st.set_page_config(page_title='Dr Sajad AI Bot')

# Hiding the header and footer
hide_menu_style="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_menu_style, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r'model.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Periapical  Xray Classification
         """
         )

file = st.file_uploader("Please upload an Periapical Xray", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.BILINEAR)
    image = np.asarray(image)
  

    # Ensure the image is grayscale
    if image.ndim == 2:  # Image is grayscale
        # Convert grayscale to RGB by duplicating the single channel
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 1:  # Image has a single channel
        # Convert single channel to RGB
        image = np.concatenate([image] * 3, axis=-1)

    # Normalize the image
    image = image / 255.0

    img_reshape = image[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)

    class_names = ['Primary Endo with Secondary Perio', 'Primary Endodontic Lesion',
                   'Primary Perio with Secondary Endo', 'Primary Periodontal Lesion', 'True Combined Lesions']
    srings = "The Lesion detected is :" + class_names[np.argmax(predictions)]

    st.text(srings)
