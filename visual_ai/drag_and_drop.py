import datarobot as dr
import pandas as pd
import matplotlib.pyplot as plt
from datarobot.models.visualai import SampleImage, ImageActivationMap, ImageEmbedding
import io
import PIL.Image

import base64
import pandas as pd
from io import BytesIO 
from PIL import Image

import streamlit as st

st.set_page_config(page_title='Visual AI Image Drag and Drop')

col1, col2 = st.columns([4,1])
with col1:
    st.title('Visual AI Image Drag and Drop')
with col2:
    st.image('dr_logo.png', width=200)

# dr.Client(config_path = "/Users/igor.veksler/.config/datarobot/drconfig.yaml")
dr.Client(endpoint = st.secrets['endpoint'], token=st.secrets['token'])


# https://docs.datarobot.com/en/docs/modeling/special-workflows/visual-ai/vai-predictions.html
def image_to_base64(image: Image) -> str:
    img_bytes = BytesIO()
    image.save(img_bytes, 'jpeg', quality=90)
    image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return image_base64

IMAGE_COLUMN = "image" # image column
TARGET_CLASS = "class" # target class
SAMPLE_SIZE = 2 # number of sample size for image sampling and activation maps

project = dr.Project.get(project_id='621fd63b2c539c5a19102e54')

# model = dr.ModelRecommendation.get(project.id) # Best method if 
#model = project.get_models[0]
model = dr.Model.get(project = project.id, model_id = '621fd9822ce435082a9578d1')

img_data = st.file_uploader(label='Load Flower Image for Detection', type=['png', 'jpg', 'jpeg'])

if img_data is not None:
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img)

    image_base64 = image_to_base64(uploaded_img)

    df = pd.DataFrame({'image': [image_base64]})
    prediction_data = project.upload_dataset(df)
    predict_job = model.request_predictions(prediction_data.id)
    result = predict_job.get_result_when_complete()
    
    st.write(result)
    

st.write('**TRAINING EXAMPLES**')
for sample in SampleImage.list(project.id, IMAGE_COLUMN)[:SAMPLE_SIZE]:
    st.write("target value = {}".format(sample.target_value))
    bio = io.BytesIO(sample.image.image_bytes)
    img = PIL.Image.open(bio)
    st.image(img)	
    #display(img)

st.write('Heatmap')
for model_id, feature_name in ImageActivationMap.models(project.id):
	for amap in ImageActivationMap.list(project.id, model_id, IMAGE_COLUMN)[:SAMPLE_SIZE]:
		bio = io.BytesIO(amap.overlay_image.image_bytes)
		img = PIL.Image.open(bio)
		st.image(img) #display(img)


