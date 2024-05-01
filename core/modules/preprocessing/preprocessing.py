from core.modules.base import BaseModule
from core.utils.preprocessing import clean_text, tokenizer
from core.database.database import Database
from core.models.accent_restore_model import AccentRestoreModel
from core.config.config import get_config
class PreprocessingModule(BaseModule):
    def __init__(self, database: Database, check_accent: bool = False, accent_path:str = None, threshold=0.5) -> None:
        self.database = database
        self.check_accent = check_accent or accent_path is not None

        if self.check_accent:
            if accent_path is None:
                raise ValueError("Accent path is required")
            
            self.accent_model = AccentRestoreModel(
                path = accent_path,
                verbose=get_config("Debug", "verbose")
            )

        self.threshold = threshold
        
        self.name = "Preprocessing"

    def _forward(self, **kwargs) -> dict:
        if self.check_accent:
            text = clean_text(
                kwargs["text"],
                synonyms_dictionary=self.database.synonyms_dictionary, 
                tokenizer=False
            )
            if self.accent_model.need_restore(text, threshold=self.threshold):
                text = self.accent_model.predict(text)
                
            text = tokenizer(text)
        else:
            text = clean_text(
                kwargs["text"], 
                synonyms_dictionary=self.database.synonyms_dictionary, 
                tokenizer=True
            )

        return {
            "text": text
        }