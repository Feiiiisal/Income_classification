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
pipeline_path = os.path.join(SRC, 'pipeline.pkl')
with open(pipeline_path, 'rb') as file:
    pipeline = pickle.load(file)

# Load the encoder using pickle
model_path = os.path.join(SRC, 'rfc_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

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
        # Convert input data to DataFrame
        input_df = pd.DataFrame([dict(income)])

        # Preprocess the input data through the pipeline
        input_df_transformed = pipeline.transform(input_df)

        # Make predictions
        prediction = model.predict(input_df_transformed)
        probability = model.predict_proba(input_df_transformed).max(axis=1)[0]

        prediction_result = "Above Limit" if prediction[0] == 1 else "Below Limit"
        return {"income_prediction": prediction_result, "prediction_probability": probability}

    except Exception as e:
        error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Error during classification: {error_detail}")


if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)
    