from lingua import Language, LanguageDetectorBuilder

languages = [Language.ENGLISH, Language.VIETNAMESE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

