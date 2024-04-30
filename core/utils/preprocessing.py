import string
import regex as re
from pyvi import ViTokenizer
import emot
emot_obj = emot.core.emot() 
import unicodedata

def normalize(word):
    word = str(word)
    word = word.strip().lower()
    word = re.sub(r"\s+", " ", word).strip()
    return word

# replace synonym word
def replace_synonym(text, synonyms_dictionary):
    text = str(text)
    for key, value in synonyms_dictionary.items():
        
        if str(key) == "nan":
            continue
        
        for keyword in str(value).split(","):
            
            if normalize(keyword).strip() != "":
                pattern = r"\b" + normalize(keyword.lower()) + r"\b"
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
    text = str(text)

    emoji_objects = emot_obj.emoji(text)
    emoticon_objects = emot_obj.emoticons(text)

    emoji_objects.update(emoticon_objects)
    for loc in emoji_objects['location']:
        text = text[:loc[0]] + "@" * (loc[1] - loc[0]) + text[loc[1]:]

    text = re.sub(r"@+", " ", text)
    text = re.sub("\s+", " ", text)
    
    return text.strip()

def clean_text_without_punctuation(text):
    text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    text = re.sub("\\s+", " ", text)
    return text

def remove_duplate_word(text):
    text = str(text)
    new_sent = []
    for word in text.split(" "):
        if len(new_sent) == 0:
            new_sent.append(word)
        else:
            if word != new_sent[-1]:
                new_sent.append(word)
    return " ".join(new_sent)

# spelling correction
def spelling_correction(text):
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
    for key,value in dictionary.items():
        text = text.replace(key,value)
    return text

stopwords = ["ak","ạ","hả","á", "à", "ấy","trôi","cậu","có thể","cậu","mình","ai đó",\
            "vậy","nhé","nha","nhỉ","mình", "vậy", "đó"]
def remove_stopword(text):
    text = str(text)
    s = ""
    for item in text.split(" "):
        if item.lower().strip() not in stopwords:
            s += item + " "
    s = re.sub("\\s+", " ",s).strip()
    return s

def tokenizer(text):
    return ViTokenizer.tokenize(text)

def clean_text(text, synonyms_dictionary, tokenizer = True):
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

    if len(text.split(" ")) > 2 and tokenizer:
        text = ViTokenizer.tokenize(text)
        
    text = text.replace("giátiền", "giá_tiền")
    return text

if __name__ == "__main__":
    text = ":)) testin :> . to , las"
    print(clean_text(text, {}, tokenizer = False))