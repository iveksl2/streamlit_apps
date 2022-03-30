import base64
from typing import Dict
import requests
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
from typing import List, Dict


from project_metadata import DEPLOYMENT_ID, API_URL, API_KEY, DATAROBOT_KEY, IMAGE_RESIZED_HEIGHT, IMAGE_RESIZED_WIDTH, IMAGE_COLUMN_NAME

def make_prediction(data: str) -> List[Dict]:
    """Uses the Prediction server to return class labels & probabilities given string representation of image

    Args:
        data (str): String representation of the Image DataFrame

    Returns:
        List[Dict]: List of Dictionaries with returned class predictions for every image
    """
    headers = {
        'Content-Type': 'text/plain; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
        }
    url = API_URL
    res = requests.post(
         url,
         data=data,
         headers=headers,
    )
    return res.json()['data']



def image_to_base64(image: Image) -> str:
    """Convert a image to base64 text format for DataRobot
    https://docs.datarobot.com/en/docs/modeling/special-workflows/visual-ai/vai-predictions.html#deep-dive

    Args:
        image (Image): jpeg image

    Returns:
        str: base64 text encoding of image
    """
    img_bytes = BytesIO()
    image.save(img_bytes, 'jpeg', quality=90)
    image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return image_base64

st.image('dr_logo.jpg', width=175)

st.title('Plant disease prediction')
uploaded_img = st.file_uploader("Upload image")

if uploaded_img is not None:
    img = Image.open(uploaded_img)
    
    img_resized = img.resize((IMAGE_RESIZED_WIDTH,IMAGE_RESIZED_HEIGHT), Image.ANTIALIAS)
    st.image(img_resized, "Uploaded image")
    st.write("Note: *This image has been resized to match training image dimensions*")

    b64_img = image_to_base64(img_resized)
    df = pd.DataFrame({IMAGE_COLUMN_NAME: [b64_img]}) 
    
    class_predictions = make_prediction(df.to_string(index=False))
    
    #todo: give this more thought
    best_pred = max(class_predictions[0]['predictionValues'], key=lambda pred: pred['value'])
    st.metric('Result',best_pred['label'],best_pred['value'])
    
    
    


