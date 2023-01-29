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
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias="native-country")
    age: int
    fnlgt: int
    education_num: int = Field(alias="education-num")
    capital_gain: int = Field(alias="capital-gain")
    capital_loss : int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")

app = FastAPI(
    title="Census Bureau data API",
    description="An API that demonstrates inferencing the census data to predict salary category",
    version="1.0.0",

)

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/inference")
async def inference(inferencedata: InferenceData = Body(
    examples={
        "normal": {
            "summary": "A normal example",
            "description": "A **normal** inrefence data works correctly",
            "value": {
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
                "age": 39,
                "fnlgt": 77516,
                "education-num": 13,
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40
            },
        },
        "invalid": {
            "summary": "Invalid example",
            "description": "An **invalid** inrefence data is rejected with an error",
            "value": {
                "workclass": 1,
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-countr": "United-States",
                "age": 39,
                "fnlgt": 77516,
                "education-num": 13,
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40
            },
        },
    },
)):

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