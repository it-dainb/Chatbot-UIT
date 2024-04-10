import re, string
from pyvi import ViTokenizer, ViPosTagger
import pandas as pd
import emoji
from pyvi import ViUtils
import json
# import check_similarity
# lowercase words
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
def give_emoji_free_text(text):
    text = str(text)
    allchars = [char for char in text.decode('utf-8')]
    emoji_list = [char for char in allchars if char in emoji.UNICODE_EMOJI]
    
    clean_text = ' '.join([word for word in text.decode('utf-8').split() if not any(emoji in word for emoji in emoji_list)])
    return clean_text

# clean text to check
def clean_text(text):
    text = str(text).replace('_', " ").lower().strip()
    return text

#THEM
# clean text to check
def clean_text_without_punctuation(text):
    text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    text = re.sub("\\s+", " ", text)
    return text

# remove icon
def remove_icon(text):
    text = str(text)
    text = text.lower()
    s = ''
    pattern = r"[a-zA-Z0123456789aăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ,._]"

    for char in text:
        if char != ' ':
            if len(re.findall(pattern, char)) != 0:
                s += char
            elif char == '_':
                s += char
        else:
            s += char
    s = re.sub('\\s+', ' ', s).strip()
    return s.strip()


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


def is_add_accent(text,threshold_add_accent=0.5):
    text = str(text)
    # full not-accent
    if len(text.strip().split(" ")) >= 2:
        remove_accents_text = str(ViUtils.remove_accents(text))[2:-1]
        if text.strip() == remove_accents_text.strip():
            return True

        # half accent
        tokens = text.strip().split(" ")
        count = 0
        for token in tokens:
          remove_accents_token = str(ViUtils.remove_accents(token))[2:-1]
          if token.strip() == remove_accents_token.strip():
              count +=1
        if len(tokens) >= 10:
            if count/len(tokens) > 0.3:
                return True
        else:
            if count / len(tokens) > threshold_add_accent:
                return True
    return False

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


# Apply pre-processing
def processing(text,synonyms_dictionary,check_accent=False):
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
    if len(text.split(" ")) > 2:
        text = ViTokenizer.tokenize(text)
    return text

#EDIT_CODE: REMOVE replace_proper_noun
# Apply pre-processing
def processing_train(text,synonyms_dictionary,check_accent=False):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text).lower()
    text = replace_synonym(text,synonyms_dictionary)
    text = remove_stopword(text)
    text = remove_icon(text)
    text = re.sub(r'\s+', " ", text).lower()
    text = spelling_correction(text)
    text = remove_duplate_word(text)
    text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    text = re.sub("\\s+", " ", text).strip()
    if len(text.split(" ")) > 2:
        text = ViTokenizer.tokenize(text)
    text = text.replace("giátiền", "giá_tiền")
    return text


if __name__ == '__main__':
    from read_database import *
    synonyms_dictionary = read_share_knowledge(path_share="Data/Data.xlsx")
    print(processing("trường mình tuyển ngành nào?",synonyms_dictionary))
