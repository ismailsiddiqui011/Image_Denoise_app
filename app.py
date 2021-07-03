import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img 
from tensorflow.keras.models import load_model
import os
from skimage.io import imread
from skimage.transform import resize
from tensorflow.image import psnr
from tensorflow.keras.optimizers import Adam
import EAM_layer
from PIL import Image

def PSNR(y_true, y_pred): 
  score = tf.image.psnr(y_true, y_pred, 1)
  return score
n = 2
model = load_model('model_3.h5', compile = False, custom_objects = {'EAM' : EAM_layer.eam_layer()})
opt = Adam(0.0001)
model.compile(opt, loss = 'mae', metrics = psnr)

st.title('Image Denoiser')
st.image('https://c.vanceai.com/assets/images/denoise_ai/denoise_mobile.jpg', width = 500)

sigma = st.text_input('Enter the Noise level (Sigma)...')
try:
  sigma = float(sigma)
except:
  st.markdown('Not a valid value, using 0 as value...')
  sigma = 0

choice = st.selectbox('Choose one of the following', ('URL', 'Upload Image'))
try:
  if choice == 'URL':
    image_path = st.text_input('Enter image URL...')
    try:
      img = imread(image_path)
      img = resize(img, (256, 256))
    except:
      st.markdown('Enter a URL')

  if choice == 'Upload Image':
    img = st.file_uploader('Upload an Image')
    if img == None:
      st.markdown('Upload Image')
    else:
      img = Image.open(img)
      img = np.array(img)/255

  n = 3
  noise = np.random.normal(scale = sigma, size = (img.shape))  
  img = img + noise
  img = np.clip(img, 0, 1)

  pred = model.predict(np.expand_dims(img, 0))[0]
  pred = np.clip(pred, 0, 1)
  st.image([img, pred], caption = ['Input', 'Prediction'], width = 256)
except:
  pass
