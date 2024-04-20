from core.database.database import Database
from core.config.config import get_config
from core.models.cls_model import ClsModel, LossType

database = Database(
    path_indomain=get_config("Path", "indomain"),
    path_outdomain=get_config("Path", "outdomain")
)

intent_model = ClsModel()
intent_data = database.create_train_label_data()
intent_model.train(intent_data, name=get_config("Model", "intent"), save_path=get_config("Path", "model"), loss=LossType.categorical)

inout_model = ClsModel()
inout_data = database.create_train_in_out_data()
inout_model.train(inout_data, name=get_config("Model", "inout"), save_path=get_config("Path", "model"), loss=LossType.binary)