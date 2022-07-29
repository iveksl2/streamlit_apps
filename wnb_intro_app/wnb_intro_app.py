# Taken from -> https://keras.io/examples/vision/grad_cam/

import altair as alt
import base64
import os
from typing import Dict
import requests
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
#from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import wandb

#wandb.init(project = 'Streamlit_Images')

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def get_img_array2(img, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4, caption='tmp'):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    #randb.log({'img2': [wandb.Image(cam_path, caption=caption)]})

    # Display Grad CAM
    #display(Image(cam_path))
    st.image(cam_path)
    return superimposed_img 

def save_and_display_gradcam2(img, heatmap, cam_path="cam.jpg", alpha=0.4, caption='tmp'):
    # Load the original image
    img = keras.preprocessing.image.load_img(img)
    img = keras.preprocessing.image.img_to_array(img)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    #randb.log({'img2': [wandb.Image(cam_path, caption=caption)]})

    # Display Grad CAM
    #display(Image(cam_path))
    st.image(cam_path)
    return superimposed_img 


image_path = os.path.join(os.path.dirname(__file__), "wnb_logo.jpg")
#image_path = "wnb_logo.jpg"
st.image(image_path, width=175)

st.title("Image Scoring and Activation Maps")

uploaded_img = st.file_uploader("Upload Image")

model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer_name = "block14_sepconv2_act"

## The local path to our target image
#img_path = keras.utils.get_file(
#    "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
#)

# https://blog.streamlit.io/session-state-for-streamlit/
if 'count' not in st.session_state:
    st.session_state.count = 0

if 'logged_data' not in st.session_state:
    st.session_state.logged_data = []

if uploaded_img is not None:

    #st.write(st.session_state.count)
    st.session_state.count +=1 

    opened_img = Image.open(uploaded_img)
    st.write('Original Image:')
    st.image(opened_img)

    # this youtube video was cash money -> https://www.youtube.com/watch?v=21y14JbQo8A
    #img_path =  f'{uploaded_img.name}'


    img_path = os.path.join(os.path.dirname(__file__), f'{uploaded_img.name}') # tmp change

    # Streamlit reads the image into RAM, but need a filepath
    opened_img.save(img_path)

    # For debugging
    #st.write(img_path)
    #st.write(os.getcwd())

    img_array = preprocess_input(get_img_array(img_path, size= img_size))
    #img_array = preprocess_input(get_img_array2(uploaded_img, size= img_size))

	# Make model
    model = model_builder(weights= "imagenet")

	# Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    preds = model.predict(img_array)
    top_predictions = decode_predictions(preds, top=2)[0]
    st.write("Top 2 Predicted Classes:", top_predictions )
    #wandb.log({f"{uploaded_img.name}_predictions": top_predictions}) 

    st.write('Activation Heatmap:')
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # tmp change
    #cam_path = save_and_display_gradcam(uploaded_img, heatmap, caption = 'elephant_activation')
    cam_path = save_and_display_gradcam(img_path, heatmap, caption = 'elephant_activation')
    #cam_path = save_and_display_gradcam2(uploaded_img, heatmap, caption = 'elephant_activation')

    #st.session_state.logged_data.append([wandb.Image(opened_img), top_predictions[0][1], wandb.Image(cam_path)])
    #st.write(st.session_state.logged_data)
    #st.write(st.session_state.count)

    if st.session_state.count%3==0 and st.session_state.count!=0:
        # Commenting out W&B Logging for the public App to not overstress the system
        #st.write('Logging WnB')
        logged_columns=['Base_Image', 'Class_Prediction',  'Activation_Map']
        #test_table = wandb.Table(data=st.session_state.logged_data, columns = logged_columns)
        #wandb.log({'new_table_name': test_table})
