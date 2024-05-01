from core.modules.base import BaseModule
from core.models.cls_model import ClsModel
from core.config.config import get_config
class ClsModule(BaseModule):
    def __init__(self, model_inout: str, model_intent: str, verbose = None):
        if verbose is None:
            verbose = get_config("Debug", "verbose")
        
        self.model_inout: ClsModel = ClsModel(model_inout, verbose)
        self.model_intent: ClsModel = ClsModel(model_intent, verbose)
        
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