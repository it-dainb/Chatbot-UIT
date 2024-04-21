from core.modules import PreprocessingModule, ClsModule
from core.database.database import Database
from core.config.config import get_config
import os
import time


database = Database(
    path_indomain=get_config("Path", "indomain"),
    path_outdomain=get_config("Path", "outdomain")
)

preprocessing = PreprocessingModule(database=database, check_accent=True)
intent = ClsModule(path=os.path.join(get_config("Path", "model"), get_config("Model", "intent")))
inout = ClsModule(path=os.path.join(get_config("Path", "model"), get_config("Model", "inout")))


pipeline = [preprocessing, inout, intent]

while(True):
    query = input("Nhập truy vấn: ")
    data = {
        "text": query
    }
    for module in pipeline:
        start = time.time()
        data = module.forward(**data)
        print(data)
        print("Time Inference: ", time.time() - start)