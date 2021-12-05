import pandas as pd
import numpy as np
import requests

if __name__ == "__main__":
    data = {'input' : ["что ты знаешь", "найди мне"]}


    response = requests.get(
        "http://127.0.0.1:8000/transform",
        json=data,
    )

    print(response.status_code)
    features = eval(response.content.decode('utf-8'))
    print(features)

