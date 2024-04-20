from core.modules.base import BaseModule
from core.utils.preprocessing import clean_text
from core.database.database import Database

class PreprocessingModule(BaseModule):
    def __init__(self, database: Database, check_accent: bool = False) -> None:
        self.database = database
        self.check_accent = check_accent

        self.name = "Preprocessing"

    def _forward(self, **kwargs) -> dict:
        return {
            "text": clean_text(
                kwargs["text"], 
                synonyms_dictionary=self.database.synonyms_dictionary, 
                check_accent=self.check_accent
            )
        }