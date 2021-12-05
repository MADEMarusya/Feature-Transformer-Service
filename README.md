# Сервис генерации признаков
Для работы сервиса используется модуль https://github.com/MADEMarusya/feature_transformator

Поэтому сначала нужно выполнить следующие действия
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit[all]
pip install spacy_udpipe
pip install numerizer
pip install transformers
pip install fasttext
pip install -r requirements.txt

# клонирование модуля генерации признаков
git clone https://github.com/MADEMarusya/feature_transformator.git
# установка модели синтактического анализа от deeppavlov
python -m deeppavlov install syntax_ru_syntagrus_bert
# Установка модели английского нумеризатора
python -m spacy download en_core_web_sm
# скачивание модель языковой идентификации 
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# Скачивание модели для пунктуации капитализации (модуль gdown указан в requirements)
gdown --id 1-1Usk7sM1aydyZFEyTetaY7tydwoG1dC
```

После подготовки можно запустить сервис
```
python app.py
```

Загрузка пайплайна перед началом работы занимает порядка 6 минут :(

Пример запроса в файле make_request.py
```
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

```

Результат запроса - строка, из которой можно создать словарь с помощью метода eval()
