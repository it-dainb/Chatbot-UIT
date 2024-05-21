from core.modules.base import BaseModule
from core.utils.preprocessing import clean_text, tokenizer
from core.database.database import Database
from core.models.accent_restore_model import AccentRestoreModel
from core.config.config import get_config
class PreprocessingModule(BaseModule):
    def __init__(self, database: Database, check_accent: bool = False, accent_path:str = None, threshold=0.5) -> None:
        """
         @brief Initialize preprocessing. This is the entry point for the restoration process. It should be called before any other method in this class.
         @param database The database to preprocess. If this is None no preprocessing will be performed.
         @param check_accent Whether to check accent prior to preprocessing. Defaults to False. See : py : class : ` accentrestore. Database ` for details.
         @param accent_path The path to the accent restore model.
         @param threshold The threshold for the restore.
         @return A : py : class : ` ~pyskool. preprocessing. Preprocessing ` object
        """
        self.database = database
        self.check_accent = check_accent or accent_path is not None

        # Restore the accent model.
        if self.check_accent:
            # If accent_path is None raise ValueError.
            if accent_path is None:
                raise ValueError("Accent path is required")
            
            self.accent_model = AccentRestoreModel(
                path = accent_path,
                verbose=get_config("Debug", "verbose")
            )

        self.threshold = threshold
        
        self.name = "Preprocessing"

    async def _forward(self, **kwargs) -> dict:
        """
         @brief Forward function for this model. This is called by the : py : meth : ` ~gensim. models. TextCorpus. forward ` method when it is called.
         @return The result of the forward function as a dictionary with the key " text " containing the text that was
        """
        # Check if the text is valid and if the user has a good score.
        if self.check_accent:
            text = clean_text(
                kwargs["text"],
                synonyms_dictionary=self.database.synonyms_dictionary, 
                tokenizer=False
            )
            # If the text needs to be restored.
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