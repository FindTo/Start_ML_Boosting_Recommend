import os
from learn_model import learn_model
from get_features_table import TYPE_MAP, csv_to_sql, load_features
from get_predict_by_model import  check_model_locally
from dotenv import load_dotenv

load_dotenv()

model, df, cat_columns = learn_model(1024000)

csv_to_sql("df_to_learn.csv",
           os.getenv('FEATURES_DF_NAME'),
           TYPE_MAP,
           int(os.getenv('CHUNKSIZE'))
           )

check_model_locally()

