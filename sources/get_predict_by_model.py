import pandas as pd
from catboost import CatBoostClassifier
from learn_model import calculate_hitrate
import os

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model_final_proj")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def check_model_locally(n: int = 1000):

    from_file = CatBoostClassifier()
    from_file.load_model(os.getenv('MODEL_NAME'))

    data = pd.read_csv('df_to_learn.csv', sep=';')

    df = data.sample(n)
    X = df.drop(['user_id', 'target', 'post_id'], axis=1)
    y = df.target

    print(from_file.predict_proba(X)[:, 1])

    hitrate = calculate_hitrate(y.values, from_file.predict_proba(X)[:, 1], k=5)
    print(f'Local hitrate for boosting model: {hitrate}')

    return 0

