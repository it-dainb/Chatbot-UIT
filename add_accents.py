# -*- coding: utf-8 -*-
import re
import unidecode , collections, itertools
from collections import Counter
from tensorflow import keras
import time
from preprocessing import *
from read_database import *
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from nltk import ngrams
class AccentRestore:
    def __init__(self, path_model, NGRAM):
        self.path_model = path_model
        with tf.device('/cpu:0'):
            self.model = keras.models.load_model(path_model,compile=False)
        self.NGRAM = NGRAM
        self.MAXLEN = self.NGRAM*6
        self.alphabet = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'

        self.accented_chars_vietnamese = [
        'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
        'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
        'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
        'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
        'í', 'ì', 'ỉ', 'ĩ', 'ị',
        'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
        'đ',
        ]
        self.alphabet = list(('\x00 ' + string.ascii_lowercase + string.digits + ''.join(self.accented_chars_vietnamese)))

    def encode(self, text):
        text = "\x00" + text
        x = np.zeros((self.MAXLEN, len(self.alphabet)))
        for i, c in enumerate(text[:self.MAXLEN]):
            x[i, self.alphabet.index(c)] = 1
        if i < self.MAXLEN - 1:
            for j in range(i+1, self.MAXLEN):
                x[j, 0] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.alphabet[i] for i in x)

    def extract_phrases(self, text):
        pattern = r'\w[\w ]*|\s\W+|\W+'
        return re.findall(pattern, text)

    def remove_accent(self, text):
        return unidecode.unidecode(text)

    def guess(self, ngram):
        text = ' '.join(ngram)
        if len(text) > self.MAXLEN:
            return text
        preds = self.model.predict(np.array([self.encode(text)]), verbose=0)
        return self.decode(preds[0], calc_argmax=True).strip('\x00')

    def gen_ngrams(self, text):
        return ngrams(text.split(), self.NGRAM)
    def add_padding(self, words):
        lenght = len(words)
        num_pad = 0
        if lenght < self.NGRAM:
            num_pad = self.NGRAM - lenght
        words = ['\x00']*num_pad + words
        return ' '.join(words)
    def add_accent(self, text):
        words = text.split(' ')
        lenght = len(words)
        if lenght < self.NGRAM:
            text = self.add_padding(words)
        ngrams = list(self.gen_ngrams(text.lower()))
        guessed_ngrams = list(self.guess(ngram) for ngram in ngrams)
        if len(guessed_ngrams) == 1:
            return ' '.join(guessed_ngrams[0].split(' ')[self.NGRAM -lenght:]).strip()
        candidates = [Counter() for _ in range(len(guessed_ngrams) + self.NGRAM - 1)]
        for nid, ngram in enumerate(guessed_ngrams):
            for wid, word in enumerate(re.split(' +', ngram)):
                candidates[nid + wid].update([word])
        output = ' '.join(c.most_common(1)[0][0] for c in candidates)
        return output
    
    def check_have_accent(self, sentence, threshhold = 0.5):
        # if len(sentence.split())<5:
        #     return False
        sentence_non_accent = self.remove_accent(sentence)
        if sentence.strip() == sentence_non_accent.strip():
            return True
        count = 0
        lst_words =  sentence.split()
        lst_words_no_accent = sentence_non_accent.split()
        for index, word in enumerate(lst_words):
            if word == lst_words_no_accent[index]:
                count+=1
        if count/len(lst_words) > threshhold:
            return True
        return False
if __name__ == "__main__":
    # path_to_model_accent_restore.h5, chọn model có NGRAM phù hợp
    NGRAM = 5
    path_model = './pickle_folder/bilstm_5gram_fulldata.h5'
    accent_restore = AccentRestore(path_model, NGRAM = NGRAM)
    textes = ["xin chao", "xin", "chao","khoa hoc may", "khoa hoc may tinh","gioi thiệu khoa hoc may",\
        "gioi thieu ve khoa hoc may tinh", "toiiiiiii muonnnnnn mccsjnsjcsncnsncnjsncncsj ncs njsnsc",\
        "toiiiiiiiiiiiiiiiiiii muonnnnnnnnnnnnnnnnnnnnnnnnnn", "ai tạo ra bot"]
    for text in textes:
        if accent_restore.check_have_accent(text):
            # print("Start restore accents...!!!")
            time_start = time.time()
            text = accent_restore.remove_accent(text)
            sentence_restored = accent_restore.add_accent(text)
            time_end = time.time()
            # print("Câu gốc:", text)
            # print("Câu đã thêm dấu:", sentence_restored)
            # print("Time inference: ", time_end - time_start)