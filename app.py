import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import requests 
from io import BytesIO
from keras.preprocessing.image import img_to_array,load_img
import os,glob,random

st.set_option("deprecation.showfileUploaderEncoding",False)
st.title("Stranger Things Character Classification")
#st.text("Provide url of cast from Stranger Things")

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./models')
	return model

with st.spinner("Loading Model Into Memory..."):
	model = load_model()

def predict_image(image_name):
	test_img = load_img(image_name,target_size=(150,150))
	img_tensor = img_to_array(test_img)
	img_tensor = np.expand_dims(img_tensor,axis=0)
	img_tensor /=255.
	pred = model.predict(img_tensor)
	result = np.argmax(pred)
	st.write(classes[result])

classes = ['Dustin','Eleven','Jonathan','Lucas','Max','Mike','Nancy','Robin','Steve','Will']


st.title("Upload + Classification Example")

#list_of_files = glob.glob('./data/*')
#first_file = max(list_of_files, key=os.path.getctime)
#st.write(latest_file)


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	img_name = "test_img.jpg"
	image.save("./data/"+str(img_name))
	st.image("./data/"+str(img_name), caption='Uploaded Image.', use_column_width=True)
	st.write("")
	st.write("Classifying...")
	predict_image("./data/"+str(img_name))

st.button(label="Refresh")


