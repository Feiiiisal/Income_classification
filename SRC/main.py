from fastapi import FastAPI,HTTPException
from typing import Literal,List
import uvicorn
from pydantic import BaseModel
import pandas as pd
import os
import pickle

# setup
SRC = os.path.abspath('./SRC/Assets')

# Load the pipeline using pickle
pipeline_path = os.path.join(SRC, 'rfc_pipeline.pkl')
with open(pipeline_path, 'rb') as file:
    rfc_pipeline = pickle.load(file)

# Load the encoder using pickle
encoder_path = os.path.join(SRC, 'encoder.pkl')
with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)

app = FastAPI(
    title= 'Income Classification FastAPI',
    description='A FastAPI service to classify individuals based on income level using a trained machine learning model.',
    version= '1.0.0'
)

class IncomePredictionInput(BaseModel):
    age:                   int
    gender:                str
    education:             str
    worker_class:          str
    marital_status:        str
    race:                  str
    is_hispanic:           str
    employment_commitment: str
    employment_stat:       int
    wage_per_hour:         int
    working_week_per_year: int
    industry_code:         int
    industry_code_main:    str
    occupation_code:       int
    occupation_code_main:  str
    total_employed:        int
    household_summary:     str
    vet_benefit:           int
    tax_status:            str
    gains:                 int
    losses:                int
    stocks_status:         int
    citizenship:           str
    importance_of_record:  float

   
class IncomePredictionOutput(BaseModel):
    income_prediction: str
    prediction_probability: float


# get
@app.get('/')
def home():
    return {
        'message': 'Income Classification FastAPI',
        'description': 'FastAPI service to classify individuals based on income level.',
        'instruction': 'Click here (/docs) to access API documentation and test endpoints.'
    }
   

# post
@app.post('/classify', response_model=IncomePredictionOutput)
def income_classification(income: IncomePredictionInput):
    try:
        df = pd.DataFrame([income.model_dump()])
           
        # Make predictions
        prediction = rfc_pipeline.predict(df)
        output = rfc_pipeline.predict_proba(df)

        prediction_result = "Income over $50K" if prediction[0] == 1 else "Income under $50K"
        return {"income_prediction": prediction_result, "prediction_probability": output[0][1]}


    except Exception as e:
        # Return error message and details if an exception occurs
        error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Error during classification: {error_detail}")


if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)
    