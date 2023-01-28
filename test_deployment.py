import requests, json

heroku_url = 'https://classification-census-data.herokuapp.com/'
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

response = requests.post(heroku_url, data=json.dumps(data))

print(response.status_code)
print(response.json())