from core.modules import BaseModule
from lingua import Language, LanguageDetectorBuilder


class LanguageModule(BaseModule):
    def __init__(self, threshold=0.95):
        """
         @brief Initialize the Languages class. This is the constructor for the Languages class. You can override this in your subclass if you want to customize the detection of languages.
         @param threshold The threshold for the confidence that a language is
        """
        
        self.languages = [Language.ENGLISH, Language.VIETNAMESE]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        self.threshold = threshold
        
        self.name = "Language"

    def is_valid_language(self, text: str) -> str:
        """
         @brief Check if the text is a valid language. This is a wrapper around the : py : meth : ` ~gensim. models. language. Detection. compute_language_confidence ` method and returns a tuple ( is_valid score ) where is_valid is True if the text is valid and False otherwise. score is the confidence in the range 0 to 1.
         @param text The text to be checked. Must be UTF - 8 encoded.
         @return A 2 - tuple of boolean and confidence. The first value is True if the text is valid. The second value is the confidence
        """
        score = self.detector.compute_language_confidence(text, language=Language.VIETNAMESE)

        # Returns true if the score is less than threshold
        if score < self.threshold:
            return False, score

        return True, score

    async def _forward(self, **kwargs):
        """
         @brief Funkcja pokud jednje tin neni jednje
         @return { language_score : } - language_score vracia pokud jednje tin nen
        """
        text = kwargs["text"]

        valid, score = self.is_valid_language(text)

        # exit with a response message.
        if not valid:
            kwargs["response"] = "Xin lỗi hiện tại tôi chỉ hỗ trợ tiếng Việt! Bạn có thắc mắc gì về việc tuyển sinh vào UIT không?"
            return self.exit(kwargs)

        return {"language_score": score}
