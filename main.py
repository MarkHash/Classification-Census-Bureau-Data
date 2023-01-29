# Put the code for your API here.
from typing import Union
import pickle
import json
import pandas as pd
import numpy as np
import yaml

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference_preds

# Declare the data object with its components and their type.
class InferenceData(BaseModel):
    workclass: str = Field(example="State-gov")
    education: str = Field(example="Bachelors")
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="Adm-clerical")
    sex: str = Field(example="Male")
    native_country: str = Field(alias="native-country", example="United-States")
    age: int = Field(example=39)
    fnlgt: int = Field(example=77516)
    education_num: int = Field(alias="education-num", example=13)
    capital_gain: int = Field(alias="capital-gain", example=2174)
    capital_loss : int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)

app = FastAPI(
    title="Census Bureau data API",
    description="An API that demonstrates inferencing the census data to predict salary category",
    version="1.0.0",

)

# This allows sending of data (our InferenceData) via POST to the API.
@app.post("/inference")
async def inference(inferencedata: InferenceData):

    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        cat_features = cfg['data']['cat_features']

    data = inferencedata.dict()
    input_data = pd.DataFrame(data=data.values(), index=data.keys()).T

    input_data = input_data.rename({"marital_status": "marital-status", 
        "native_country": "native-country", 
        "education_num": "education-num", 
        "capital_gain": "capital-gain", 
        "capital_loss": "capital-loss", 
        "hours_per_week": "hours-per-week"}, axis=1)
    
    prediction = inference_preds(input_data, cat_features)
    return {"prediction": prediction}

@app.get("/")
async def welcome_message():
    return {"message": "Welcome to the Census Bureau data project"}