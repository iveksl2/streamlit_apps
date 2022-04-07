import altair as alt
import os
import datarobot as dr
import requests
import pandas as pd
import streamlit as st
from typing import List

from project_metadata import (
    API_KEY,
    DATAROBOT_KEY,
    DATASET_ID,
    DEPLOYMENT_ID,
    ANOMALY_DEPLOYMENT_ID,
    TOKEN
)

@st.cache(allow_output_mutation=True)
def get_raw_data(dataset_id: str) -> pd.DataFrame:
    """Retrieves raw data from the AI Catalog that initiaties a project.
    https://datarobot-public-api-client.readthedocs-hosted.com/en/v2.20.0/entities/dataset.html#retrieving-datasets

    Args:
        dataset_id (str): Dataset ID. Can be retrieved from the Browser from an AI Catalog Item

    Returns:
        pd.DataFrame: Original dataset.
    """
    # TODO: FIX THIS FUNCTION.
    # It's reading full data basically as a workaround for https://datarobot.atlassian.net/browse/DSX-2141
    # Everything we do will be in sample, and this is hard to generalize to new projects
    # https://github.com/dansbecker/datarobot_churn_app/blob/main/understanding_churn_section.py
    dataset = dr.Dataset.get(dataset_id)
    dataset.get_file("tmp.csv")
    return pd.read_csv("tmp.csv").copy()


@st.cache
def get_datarobot_predictions(
    deployment_id: str, pred_data: pd.DataFrame
) -> List[float]:
    """Score a model provided a dataset using the prediction server

    Args:
        deployment_id (str): Deployment ID
        pred_data (pd.DataFrame): Scoring dataset

    Returns:
        List[float]: Model Predictions
    """
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": "Bearer {}".format(API_KEY),
        "DataRobot-Key": DATAROBOT_KEY,
    }

    url = (
        "https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/"  # hardcoded cfds deployment machine
        + deployment_id
        + "/predictions"
    )

    predictions_response = requests.post(
        url,
        data=pred_data.to_json(orient="records"),
        headers=headers,
    )
    predictions = [
        dict["predictionValues"][0]["value"]
        for dict in predictions_response.json()["data"]
    ]
    return predictions


@st.cache
def get_datarobot_prediction_explanations(
    deployment_id: str, pred_data: pd.DataFrame
) -> pd.DataFrame:
    """Retreive prediction explanations from a deployment

    Args:
        deployment_id (str): 
        pred_data (pd.DataFrame):

    Returns:
        pd.DataFrame: Each explanation is a dataframe row
    """
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": "Bearer {}".format(API_KEY),
        "DataRobot-Key": DATAROBOT_KEY,
    }

    url = (
        "https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/"  # hardcode cfds deployment machine
        + deployment_id
        + "/predictions"
    )

    params = {"maxExplanations": 3}
    predictions_response = requests.post(
        url, data=pred_data.to_json(orient="records"), headers=headers, params=params
    )
    explanations = pd.DataFrame(
        predictions_response.json()["data"][0]["predictionExplanations"]
    )
    return explanations


dr.Client(endpoint = 'https://app.datarobot.com/api/v2', token=TOKEN)
image_path = os.path.join(os.path.dirname(__file__), "dr_logo.jpg")
st.image(image_path, width=175)

st.title("Anti Money Laundering")

st.write(
    """
<h4 style='margin-top:-0px'>Description</h4>
This application is designed to assist in the adjudication of money laundering alerts.
<br>
Users can toggle criticality thresholds between DataRobot predictions and anomaly scores. 
<br>
If the checkbox is enabled, the resulting dataset is filtered to records where both scores are above or equal to the thresholds. 
<br><br>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([3, 3, 1])

with col1:
    prediction_threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
with col2:
    anomaly_threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.0, 0.01)

aml_df = get_raw_data(DATASET_ID)
aml_df["row_id"] = range(len(aml_df))
aml_df["Prediction"] = get_datarobot_predictions(DEPLOYMENT_ID, aml_df)
aml_df["anomaly_scores"] = get_datarobot_predictions(ANOMALY_DEPLOYMENT_ID, aml_df)

filter_checkbox_on = st.checkbox("Filter by Prediction & Anomaly Thresholds")

filtered_df = aml_df.copy()
if filter_checkbox_on:
    filtered_df = aml_df.query(
        f"Prediction >= {prediction_threshold} & anomaly_scores >= {anomaly_threshold}"
    )

with col3:
    st.metric("Total Alerts", filtered_df.shape[0])

st.write(filtered_df)

download_button = st.download_button(
    "Download Alerts",
    filtered_df.to_csv(index=False),
    file_name="alerts.csv",
    key="download_button",
)

row_id = st.selectbox("Select a Transaction Number (row_id) and drill down for explanations of prediction drivers", aml_df[["row_id"]])
drill_down_flag = st.button("Drill Down (Prediction Explanations)")
if drill_down_flag:
    row_data = aml_df[aml_df["row_id"] == row_id]
    row_explanations = get_datarobot_prediction_explanations(DEPLOYMENT_ID, row_data)
    st.write(
        alt.Chart(row_explanations[['feature', 'strength']]).mark_bar().encode(
                x='strength',    
                y='feature',
        ).properties(
            width=400,
            height=350,
            title='DataRobot Model Prediction Explanations'
        )
    )
    st.markdown('''This provides the drivers of the DataRobot model. <br> 
                Learn more about [DataRobot Prediction Explanations](https://docs.datarobot.com/en/docs/modeling/analyze-models/understand/pred-explain/index.html)''',  
                unsafe_allow_html=True,)

    
uploaded_file = st.file_uploader("Upload data to make a prediction. Note: Make sure to include the same column names from the training dataset (<50MB)", type="csv")

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    
    #TODO: wrap below into functions to not repeat code
    uploaded_df["row_id"] = range(len(uploaded_df))
    uploaded_df["Prediction"] = get_datarobot_predictions(DEPLOYMENT_ID, uploaded_df)
    uploaded_df["anomaly_scores"] = get_datarobot_predictions(ANOMALY_DEPLOYMENT_ID, uploaded_df)   
    
    col4, col5, col6 = st.columns([3, 3, 1])
    
    with col4:
        prediction_threshold_uploaded = st.slider("Prediction Threshold Upload", 0.0, 1.0, 0.5, 0.01)
    with col5:
        anomaly_threshold_uploaded = st.slider("Anomaly Threshold Upload", 0.0, 1.0, 0.0, 0.01)

    upload_filter_checkbox_on = st.checkbox("Filter Uploaded file by Prediction & Anomaly Thresholds")

    filtered_df_uploaded = uploaded_df.copy()
    if upload_filter_checkbox_on:
        filtered_df_uploaded = uploaded_df.query(
            f"Prediction >= {prediction_threshold_uploaded} & anomaly_scores >= {anomaly_threshold_uploaded}"
        )
    
    with col6:    
        st.metric("Total Uploaded File Alerts", filtered_df_uploaded.shape[0])
    st.write(filtered_df_uploaded)
    
    download_button = st.download_button(
    "Download Uploaded Alerts",
    filtered_df_uploaded.to_csv(index=False),
    file_name="uploaded_alerts.csv",
    key="download_button",
)
    
