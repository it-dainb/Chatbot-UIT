from core.modules.base import BaseModule
from core.models.cls_model import ClsModel
from core.config.config import get_config
class ClsModule(BaseModule):
    def __init__(self, model_inout: str, model_intent: str, verbose = None):
        """
         @brief Initialize the class. This is the constructor for the class. You need to call it yourself if you want to use the Classification class
         @param model_inout path to the input file
         @param model_intent path to the intent file ( optional )
         @param verbose whether to print debug information ( optional default : True
        """
        # If verbose is not set the verbose flag to true.
        if verbose is None:
            verbose = get_config("Debug", "verbose")
        
        self.model_inout: ClsModel = ClsModel(model_inout, verbose)
        self.model_intent: ClsModel = ClsModel(model_intent, verbose)
        
        self.name: str = "Classification"

    async def _forward(self, **kwargs):
        """
        @brief Forward function for the bot. This is called by the : meth : ` ~CorpheatBot. run ` method.
        @return a dictionary with the following keys : chat_text : the text that was sent intent : the intent that was
        """

        text = kwargs["text"]
        
        result = {
            "chat_text": text.replace("_", " ")
        }
        
        domain = await self.model_inout.predict(text)
        domain = domain[0]["class"]

        result["domain"] = domain
        result["intent"] = None
        result["text"] = text

        # in or in the domain of the intent
        if domain == "in":
            intent = await self.model_intent.predict(text)
            
            result["intent"] = intent[0]["class"]
            result["intent_score"] = intent[0]["prob"]
        
        return result