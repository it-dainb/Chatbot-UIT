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

    async def _forward(self, **kwargs):

        text = kwargs["text"]
        
        result = {
            "chat_text": text.replace("_", " ")
        }
        
        domain = await self.model_inout.predict(text)
        domain = domain[0]["class"]

        result["domain"] = domain
        result["intent"] = None
        result["text"] = text

        if domain == "in":
            intent = await self.model_intent.predict(text)
            
            result["intent"] = intent[0]["class"]
            result["intent_score"] = intent[0]["prob"]
        
        return result