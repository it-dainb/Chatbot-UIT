from core.modules.base import BaseModule
from core.models.cls_model import ClsModel

class ClsModule(BaseModule):
    def __init__(self, path: str):
        self.model: ClsModel = ClsModel(path)
        self.name: str = f"{self.model.config['name'].capitalize()} Classification"

    def _forward(self, **kwargs):
        return {
            f"{self.model.config['name']}_class": self.model.predict(kwargs["text"]),
        }