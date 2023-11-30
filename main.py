"""
Car price prediction service.
"""
from __future__ import annotations

import codecs
import csv
import io

import model as md
import pandas as pd

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float
class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
async def predict_item(item: Item):
    """
    Predict price for one car (data in json format).
    """
    # load pre-trained model
    model = md.load_model('model.pkl')

    # save json to data frame
    df_item = pd.DataFrame([item.model_dump()])

    # run data pre-processing and model prediction
    df_pre = md.data_preprocessing(df=df_item)
    predicted_price  = md.run_model(model=model, df=df_pre)

    return predicted_price['predicted_price'][0]

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    """
    Predict price for cars from csv file and return csv file with additional column predicted_price
    """
    # read data from csv and convert to data frame
    csv_reader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    df = pd.DataFrame(csv_reader)

    # convert pre-defined columns to numeric
    df[['year', 'selling_price', 'km_driven']] = df[['year', 'selling_price', 'km_driven']].apply(pd.to_numeric)
    df['seats'] = df['seats'].astype(float)

    # save original price
    original_price = df['selling_price']

    # run data pre-processing and model prediction
    df_pre = md.data_preprocessing(df=df)
    model = md.load_model('model.pkl')
    df_pre = md.run_model(model=model, df=df_pre)
    df_pre['selling_price'] = original_price

    # save data frame with prediction into csv and sand in response
    stream = io.StringIO()
    df_pre.to_csv(stream, index=False)
    file.file.close()
    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predicted.csv"
    return response

@app.get('/')
def root():
    """
    Welcome to the car price prediction service !
    :return: welcome to string in dict
    """
    return "Welcome to the cars prediction service !"