import streamlit as st
from tensorflow import keras
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
import numpy as np
import os
from pathlib import Path
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

st.title('Skin Lesion Classifier')
path = Path(__file__)
@st.cache(allow_output_mutation=True)
def loadModel():
    model_path = str(path.resolve().parents[0]) + '/model/mobilenetv3.h5'
    model_idc = keras.models.load_model(model_path,
    compile=True,
    custom_objects={'top_2_accuracy': top_2_accuracy,'top_3_accuracy': top_3_accuracy})
    return model_idc
@st.cache(allow_output_mutation=True)
def transform_image(uploaded_file,img_size=224):
    # transform image to numpy array
    image_uploaded = keras.preprocessing.image.load_img(uploaded_file, target_size=(img_size,img_size), 
        grayscale = False, interpolation = 'nearest', color_mode = 'rgb', keep_aspect_ratio = False)
    input_arr = keras.preprocessing.image.img_to_array(image_uploaded)
    input_arr = keras.applications.mobilenet_v3.preprocess_input(input_arr)

    
    return np.expand_dims(input_arr, axis=0)

container = st.container()

uploaded_file = container.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file)

Generate_pred = container.button("Predict")
if Generate_pred:
    model = loadModel()
    #image = transform_image(uploaded_file)
    prediction = model.predict(transform_image(uploaded_file))
    classes_dict = {0: 'Actinic Keratosis/Intraepithelial carcinom', 1: 'Basal Cell Carcinoma', 2: 'Actinic Keratosis', 3: 'Dermatofibroma', 4: 'Melanoma', 5: 'Melanocytic Nevus', 6: 'Vascular lesion'}
    result = classes_dict[np.argmax(prediction)]
    #container.metric('Predição', delta_color='normal')
    container.metric('Predição', result, delta_color='normal')

    
