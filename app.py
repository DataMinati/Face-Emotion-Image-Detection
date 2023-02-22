import streamlit as st
import tensorflow as tf

#st.set_option('deprecation.showFileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Facial Expression Recognition
         """
         )

file = st.file_uploader("Please upload a picture of your face..", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (48,48)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_resize = (cv2.resize(img, dsize=(48, 48)))
        
        img_reshape = img.reshape(1,48,48,1)
    
        prediction = model.predict(img_reshape)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Angry', 'Neutral', 'Scared', 'Happy', 'Sad', 'Surprised']
    score = tf.nn.softmax(predictions[0])
    #st.write(predictions)
    st.write("The person in the image is {} ".format(class_names[np.argmax(predictions)]))
