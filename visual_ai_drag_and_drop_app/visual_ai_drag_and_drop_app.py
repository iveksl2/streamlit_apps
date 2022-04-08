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



from project_metadata import (
    DEPLOYMENT_ID,
    API_URL,
    API_KEY,
    DATAROBOT_KEY,
    IMAGE_RESIZED_HEIGHT,
    IMAGE_RESIZED_WIDTH,
    IMAGE_COLUMN_NAME,
)


def make_prediction(data: str) -> pd.DataFrame:
    """Uses the Prediction server to return class labels & probabilities given string representation of image

    Args:
        data (str): String representation of the Image DataFrame

    Returns:
        #TODO: Update
        List[Dict]: List of Dictionaries with returned class predictions for every image
    """
    headers = {
        "Content-Type": "text/plain; charset=UTF-8",
        "Authorization": "Bearer {}".format(API_KEY),
        "DataRobot-Key": DATAROBOT_KEY,
    }
    url = API_URL
    res = requests.post(
        url,
        data=data,
        headers=headers,
    )
    #return res.json()["data"]
    predictions = res.json()["data"]
    pred_df = pd.DataFrame(predictions[0]['predictionValues']).sort_values(by='value', ascending=False).reset_index(drop=True)
    return pred_df


def image_to_base64(image: Image) -> str:
    """Convert a image to base64 text format for DataRobot
    https://docs.datarobot.com/en/docs/modeling/special-workflows/visual-ai/vai-predictions.html#deep-dive

    Args:
        image (Image): jpeg image

    Returns:
        str: base64 text encoding of image
    """
    img_bytes = BytesIO()
    image.save(img_bytes, "png", quality=90)
    image_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return image_base64

def prep_image_for_scoring(img: Image) -> pd.DataFrame:
    img_resized = img.resize(
        (IMAGE_RESIZED_WIDTH, IMAGE_RESIZED_HEIGHT), Image.ANTIALIAS
    )
    b64_img = image_to_base64(img_resized)
    df = pd.DataFrame({IMAGE_COLUMN_NAME: [b64_img]})
    return df.to_string(index=False)

def render_prediction_barchart(df: pd.DataFrame) -> alt.Chart:
    chrt = alt.Chart(df).mark_bar().encode(
        alt.X('value', title='Predictions'),    
        alt.Y('label', title='Class Labels', sort='-x')
    ).properties(
        width=400,
        height=350,
    )
    return chrt

def prep_score_render_output(img: Image) -> None:
    #TODO: See if method chaining is better
    img_df = prep_image_for_scoring(img)
    pred_df = make_prediction(img_df)
    chrt = render_prediction_barchart(pred_df)
    st.write("Most likly class is",  pred_df.loc[0, 'label'], ' - the model prediction is', pred_df.loc[0, 'value'])
    st.write(chrt)
    return

image_path = os.path.join(os.path.dirname(__file__), "dr_logo.jpg")
st.image(image_path, width=175)

st.title("Plant disease prediction")

uploaded_img = st.file_uploader("Upload Image")

if uploaded_img is not None:
    img = Image.open(uploaded_img)

    img_resized = img.resize(
        (IMAGE_RESIZED_WIDTH, IMAGE_RESIZED_HEIGHT), Image.ANTIALIAS
    )
    st.image(img_resized, "Uploaded image")
    st.write("Note: *This image has been resized to match training image dimensions*")

    b64_img = image_to_base64(img_resized)
    df = pd.DataFrame({IMAGE_COLUMN_NAME: [b64_img]})

    pred_df = make_prediction(df.to_string(index=False))

    #pred_df = pd.DataFrame(predictions[0]['predictionValues']).sort_values(by='value', ascending=False).reset_index(drop=True)
    
    chrt = alt.Chart(pred_df).mark_bar().encode(
            alt.X('value', title='Predictions'),    
            alt.Y('label', title='Class Labels', sort='-x')
    ).properties(
        width=400,
        height=350,
    )
    
    st.write("Most likly class is",  pred_df.loc[0, 'label'], ' - the model prediction is', pred_df.loc[0, 'value'])
    st.write(chrt)
    
st.header('Example Images')
#TODO: DRY?
col1, col2, col3 = st.columns(3)
with col1:
    img1_path = os.path.join(os.path.dirname(__file__), "images/bell_pepper_sample.JPG")
    st.image(img1_path)
    img1_button = st.button("Score Image", key='1')
with col2:
    img2_path = os.path.join(os.path.dirname(__file__), "images/tomato_bacterial_spot_sample.JPG")
    st.image(img2_path)
    img2_button = st.button("Score Image", key='2')
with col3:
    img3_path = os.path.join(os.path.dirname(__file__), "images/tomato_leafspot_sample.JPG")
    st.image(img3_path)
    img3_button = st.button("Score Image", key='3')

if img1_button:
    img1 = Image.open(img1_path)
    prep_score_render_output(img1)
elif img2_button:
    img2 = Image.open(img2_path)
    prep_score_render_output(img2)
elif img3_button:
    img3 = Image.open(img3_path)
    prep_score_render_output(img3)

    