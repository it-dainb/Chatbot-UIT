from core.modules import BaseModule
from lingua import Language, LanguageDetectorBuilder


class LanguageModule(BaseModule):
    def __init__(self, threshold=0.95):
        
        self.languages = [Language.ENGLISH, Language.VIETNAMESE]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        self.threshold = threshold
        
        self.name = "Language"

    def is_valid_language(self, text: str) -> str:
        score = self.detector.compute_language_confidence(text, language=Language.VIETNAMESE)

        if score < self.threshold:
            return False, score

        return True, score

    def _forward(self, **kwargs):
        text = kwargs["text"]

        valid, score = self.is_valid_language(text)

        if not valid:
            kwargs["response"] = "Xin lỗi hiện tại tôi chỉ hỗ trợ tiếng Việt! Bạn có thắc mắc gì về việc tuyển sinh vào UIT không?"
            return self.exit(kwargs)

        return {"language_score": score}
