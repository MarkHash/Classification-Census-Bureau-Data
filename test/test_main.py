import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census Bureau data project"}

def test_post_data_under():
    data = {"workclass": "State-gov",
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
            }
    r = client.post("/", data=json.dumps(data)) 
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}

def test_post_data_over():
    data = {"workclass": "Self-emp-not-inc",
            "education": "HS-grad",
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "native-country": "United-States",
            "age": 52,
            "fnlgt": 209642,
            "education-num": 9,
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 45
            }
    r = client.post("/", data=json.dumps(data)) 
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}

def test_post_data_fail():
    data = {"workclass": 1,
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
            "hours-per-week": 40}
    r = client.post("/", data=json.dumps(data))
    assert r.status_code == 422
