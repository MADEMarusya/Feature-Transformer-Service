import sys
sys.path.append("./feature_transformator")

import os
import pickle
import pandas as pd
import numpy as np
import json
import requests
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, conlist
from typing import List, Union, Optional, TypeVar
from feature_transformator.src.feature_pipeline import FeaturePipeline
import torch
import pandas as pd
from sklearn.pipeline import Pipeline

# Путь до модели капитализации пунктуации
PATH_TO_CAPIT_PUNKT_MODEL = './model-epoch=00-val_loss=0.0680.ckpt.nemo'

# Путь до модели языковой идентификации
PATH_TO_LID_MODEL = './lid.176.bin'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

app = FastAPI()
transfrom_pipeline: Optional[Pipeline] = None

class OutputItem(BaseModel):
    transform: dict

class Item(BaseModel):
    input: list

with open("./raw_query_flow.txt", "r", encoding='utf-8') as fin:
    rawqq = fin.readlines()
data_for_fit = pd.DataFrame(rawqq, columns=["input"])

pipeline = FeaturePipeline(
    path_to_capit_punkt_model=PATH_TO_CAPIT_PUNKT_MODEL,
    path_to_lid_model=PATH_TO_LID_MODEL,
    fit_data_for_topic_prediction=data_for_fit["input"].values,
    device=device,
)


@app.get("/")
def main():
    return "Marussia queries API"

@app.on_event("startup")
def load():
    pass

@app.get("/transform", response_model = OutputItem)
def transform(request: Item):
    data = pd.DataFrame([
        {'input': i} for i in request.dict()['input']
    ])

    return OutputItem(transform=pipeline.fit_transform(data).to_dict())

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))