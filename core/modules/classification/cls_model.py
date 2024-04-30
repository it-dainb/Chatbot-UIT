from core.modules.base import BaseModule
from core.models.cls_model import ClsModel

class ClsModule(BaseModule):
    def __init__(self, model_inout: str, model_intent: str):
        self.model_inout: ClsModel = ClsModel(model_inout)
        self.model_intent: ClsModel = ClsModel(model_intent)
        
        self.name: str = "Classification"

    def _forward(self, **kwargs):

        text = kwargs["text"]

        

        domain = self.model_inout.predict(text)[0]["class"]
        
        result = {
            "domain": domain,
            "intent": None
        }

        if domain == "in":
            intent = self.model_intent.predict(text)[0]["class"]
            result["intent"] = intent
        
        result["text"] = text.replace("_", " ")
        
        return result