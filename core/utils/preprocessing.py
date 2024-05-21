import string
import regex as re
from pyvi import ViTokenizer
import emot
emot_obj = emot.core.emot() 
import unicodedata

def normalize(word):
    """
     @brief Normalizes a word to be used as a stem. This is a case - insensitive version of the word that is lowercase with no spaces and all whitespace removed.
     @param word The word to normalize. Must be a string or unicode object.
     @return A lower case string of the word normalized to a single space. >>> normalize ('Arabic')'Arabic
    """
    word = str(word)
    word = word.strip().lower()
    word = re.sub(r"\s+", " ", word).strip()
    return word

# replace synonym word
def replace_synonym(text, synonyms_dictionary):
    """
     @brief Replaces synonyms in a text. Replaces all occurences of a synonym with the value of the synonym
     @param text text to search for synonyms
     @param synonyms_dictionary dictionary with synonyms as keys and values as values
     @return text with synonyms replaced with the value of the synonym ( s ) if they exist in the dictionary else the original
    """
    text = str(text)
    # This function takes a dictionary of synonyms and returns a string of the form key value.
    for key, value in synonyms_dictionary.items():
        
        # If the key is nan continue to use nan.
        if str(key) == "nan":
            continue
        
        # This function takes a string of keyword and returns a string of the form key value.
        for keyword in str(value).split(","):
            
            # normalize keyword and remove the \ b and \ b
            if normalize(keyword).strip() != "":
                pattern = r"\b" + normalize(keyword.lower()) + r"\b"
                # normalize key to a string.
                if "\\" in key:
                    text = re.sub(pattern, r" "+normalize(key)+ r" ", normalize(text))
                else:
                    # print(key, pattern)
                    text = re.sub(pattern, " "+normalize(key)+ " ", normalize(text))


    dup_word = r'(?<![^\s])(([^\s]+)\s)\2+'
    text = re.sub(dup_word, r'\2', normalize(text))

    return text

# remove emoji
def remove_icon(text):
    """
     @brief Removes emoji emoticons from text. This is useful for cleaning up text that is displayed to the user in a web browser.
     @param text The text to be cleaned. Should be a string of emoji or emoticons.
     @return The cleaned text after removing emoji emoticons and space characters from the text. If there is no emoji emoticons or space characters in the text they are removed
    """
    text = str(text)

    emoji_objects = emot_obj.emoji(text)
    emoticon_objects = emot_obj.emoticons(text)

    emoji_objects.update(emoticon_objects)
    # Returns the emoji s location as a string.
    for loc in emoji_objects['location']:
        text = text[:loc[0]] + "@" * (loc[1] - loc[0]) + text[loc[1]:]

    text = re.sub(r"@+", " ", text)
    text = re.sub("\s+", " ", text)
    
    return text.strip()

def clean_text_without_punctuation(text):
    """
     @brief Clean text removing punctuation and lowercasing. This is useful for converting text to human readable format.
     @param text The text to clean. Must be a string or unicode object.
     @return A string with all punctuation removed and lowercased. If you want to clean a string use clean_text
    """
    text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    text = re.sub("\\s+", " ", text)
    return text

def remove_duplate_word(text):
    """
     @brief Removes words from a text that are present in duplate. This is useful for generating an email that is sent to a user in order to check if they are in a dictionnary or not
     @param text The text to be checked
     @return The text with words removed from the dictionnary if they are present in the text otherwise a
    """
    text = str(text)
    new_sent = []
    # Add a new word to the sent list.
    for word in text.split(" "):
        # Add word to new_sent if the word is not the last word in the list.
        if len(new_sent) == 0:
            new_sent.append(word)
        else:
            # Add a new word to the sent list.
            if word != new_sent[-1]:
                new_sent.append(word)
    return " ".join(new_sent)

# spelling correction
def spelling_correction(text):
    """
     @brief Correct spelling of text. This is used to correct spelling of a text that is in an unkonwn word by replacing it with a word that is different from the original.
     @param text The text to be corrected. Should be a string.
     @return The text with spelling corrected. It is a string but not a string because the input is converted to a string
    """
    text = str(text)
    dictionary = {
        'xoá': 'xóa',
        'huỷ': 'hủy',
        'hũy': 'hủy',
        'huỹ': 'hủy',
        'xõa': 'xóa',
        'khoá': 'khóa',
        'hoá' : 'hóa',
        'lải': 'lãi',
        'suât': 'suất',
        'chuyền': 'chuyển',
        'khâu': 'khẩu',
        'khảu': 'khẩu',
        'hổ': 'hỗ',
    }
    # Replaces all the values of the key value pairs in the dictionary with their values.
    for key,value in dictionary.items():
        text = text.replace(key,value)
    return text

stopwords = ["ak","ạ","hả","á", "à", "ấy","trôi","cậu","có thể","cậu","mình","ai đó",\
            "vậy","nhé","nha","nhỉ","mình", "vậy", "đó"]
def remove_stopword(text):
    """
     @brief Removes stopwords from a string. This is useful for stripping out words that are too frequent to be considered as part of a word.
     @param text The text to remove stopwords from. Can be a string or a list of strings.
     @return The text without stopwords ( space separated ). >>> remove_stopword ('A B C D E F G H I')'A B C
    """
    text = str(text)
    s = ""
    # Returns a string of stopwords.
    for item in text.split(" "):
        # Add a stopword to the end of the string.
        if item.lower().strip() not in stopwords:
            s += item + " "
    s = re.sub("\\s+", " ",s).strip()
    return s

def tokenizer(text):
    """
     @brief Tokenize text into a list of tokens. This is a wrapper around the L { ViTokenizer. tokenize } function
     @param text The text to tokenize.
     @return A list of tokens in the form of ( token_type token_data ) where token_type is one of the constants defined in this module
    """
    return ViTokenizer.tokenize(text)

def clean_text(text, synonyms_dictionary, tokenizer = True):
    """
     @brief Clean text for use in search. This is the heart of the cleaning process. It removes stopwords icon spelling correction duplate word and other words
     @param text text to be cleaned.
     @param synonyms_dictionary dictionary of synonyms that should be used for cleanup
     @param tokenizer flag to enable / disable tokenizer
     @return cleaned text as a string of lowercase words with special characters removed and synonyms tokenized if it contains more than one
    """
    text = unicodedata.normalize("NFC", text)
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text).lower()
    text = replace_synonym(text,synonyms_dictionary)
    text = remove_stopword(text)
    text = remove_icon(text)
    text = text.strip()
    text = re.sub(r'\s+', " ", text).lower()
    text = spelling_correction(text)
    text = remove_duplate_word(text)
    text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    text = re.sub("\\s+", " ", text).lower()

    # If the text is a string of length 2 and tokenizer is set to a ViTokenizer. tokenizer.
    if len(text.split(" ")) > 2 and tokenizer:
        text = ViTokenizer.tokenize(text)
        
    text = text.replace("giátiền", "giá_tiền")
    return text

# prints the text in the text
if __name__ == "__main__":
    text = ":)) testin :> . to , las"
    print(clean_text(text, {}, tokenizer = False))