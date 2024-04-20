from core.config.config import load_config, get_config
from core.models.cls_model import ClsModel
from core.utils.preprocessing import clean_text
import os
from core.database.database import Database
import time

load_config("config.cfg")

database = Database(
    path_indomain=get_config("Path", "indomain"),
    path_outdomain=get_config("Path", "outdomain")
)

intent_model = ClsModel(path=os.path.join(get_config("Path", "model"), get_config("Model", "intent")))
inout_model = ClsModel(path=os.path.join(get_config("Path", "model"), get_config("Model", "inout")))

while(True):
    query = input("Nhập truy vấn: ")
    user_query_clean = clean_text(query, database.synonyms_dictionary)
    print("Pre-processing: ", user_query_clean)

    start = time.time()
    result = intent_model.predict(user_query_clean)
    print(result)
    print("Time Inference: ", time.time() - start)

    start = time.time()
    result = inout_model.predict(user_query_clean)
    print(result)
    print("Time Inference: ", time.time() - start)