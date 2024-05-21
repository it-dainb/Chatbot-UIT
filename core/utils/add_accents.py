# -*- coding: utf-8 -*-
import re
import unidecode , collections, itertools
from collections import Counter
from tensorflow import keras
import time
from preprocessing import *
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from nltk import ngrams

class AccentRestore:
    """
    Lớp AccentRestore dùng để khôi phục dấu trong văn bản tiếng Việt.

    Lớp này sử dụng mô hình học sâu đã được huấn luyện để dự đoán và thêm dấu vào các từ không dấu.
    Nó có thể xử lý văn bản dài và tạo ra văn bản có dấu chính xác.

    Attributes:
        path_model (str): Đường dẫn đến mô hình học sâu đã được lưu.
        NGRAM (int): Số lượng từ tối đa trong một n-gram.
        MAXLEN (int): Độ dài tối đa của chuỗi đầu vào cho mô hình.
        alphabet (list): Danh sách các ký tự được sử dụng, bao gồm cả ký tự có dấu và không dấu.
        accented_chars_vietnamese (list): Danh sách các ký tự có dấu trong tiếng Việt.

    Methods:
        encode(text): Mã hóa văn bản thành dạng one-hot encoding để đưa vào mô hình.
        decode(x, calc_argmax=True): Giải mã kết quả dự đoán từ mô hình thành văn bản có dấu.
        extract_phrases(text): Trích xuất các cụm từ từ văn bản.
        remove_accent(text): Loại bỏ dấu từ văn bản.
        guess(ngram): Dự đoán dấu cho một n-gram.
        gen_ngrams(text): Tạo ra các n-gram từ văn bản.
        add_padding(words): Thêm padding vào danh sách từ để đạt đủ độ dài NGRAM.
        add_accent(text): Thêm dấu vào văn bản không dấu.
        check_have_accent(sentence, threshold=0.5): Kiểm tra xem câu có chứa dấu hay không.
    """
    def __init__(self, path_model, NGRAM):
        """
         @brief Initializes the model and sets max length. This is called by __init__
         @param path_model path to the Keras model
         @param NGRAM number of words in the n - gram
        """
        self.path_model = path_model
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
        """
         @brief Encodes text into a 2D array where each row corresponds to a character in the alphabet.
         @param text The text to encode. Must be at least MAXLEN characters long.
         @return A 2D NumPy array of length self. MAXLEN containing 1 for each character in text
        """
        text = "\x00" + text
        x = np.zeros((self.MAXLEN, len(self.alphabet)))
        # x i len text. len x.
        for i, c in enumerate(text[:self.MAXLEN]):
            x[i, self.alphabet.index(c)] = 1
        # Set x to 0. 0x0 0x1 0x2 0x3 0x4 0x4 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0x5 0
        if i < self.MAXLEN - 1:
            # Set x 0 to the first element of x.
            for j in range(i+1, self.MAXLEN):
                x[j, 0] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """
         @brief Decode a sequence of character IDs. This is a convenience method for making use of : meth : ` ~gensim. models. Characters. alphabet ` to convert a sequence of character IDs into a string of characters.
         @param x A numpy array of character IDs. The length of this array must be the same as the number of characters in the alphabet
         @param calc_argmax If
        """
        # Calculate the maximal value of the axis.
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.alphabet[i] for i in x)

    def extract_phrases(self, text):
        """
         @brief Extract phrases from text. This is used to extract phrases that are part of a word or phrase list.
         @param text text to extract phrases from. Must be a string
         @return list of strings each of which is a
        """
        pattern = r'\w[\w ]*|\s\W+|\W+'
        return re.findall(pattern, text)

    def remove_accent(self, text):
        """
         @brief Remove accented characters from text. This is useful for translating text that is in a text field such as text_for_person and vice versa.
         @param text The text to unidecode. Must be unicode.
         @return The text with accented characters removed. If the text is not unicode it will be returned as is
        """
        return unidecode.unidecode(text)

    def guess(self, ngram):
        """
         @brief Guess a text based on a list of n - grams. This is a convenience method for use when you don't know what you're doing.
         @param ngram A list of n - grams.
         @return A guess for the text or the unigram
        """
        text = ' '.join(ngram)
        # Returns the text if the length of the text exceeds the maximum allowed length.
        if len(text) > self.MAXLEN:
            return text
        preds = self.model.predict(np.array([self.encode(text)]), verbose=0)
        return self.decode(preds[0], calc_argmax=True).strip('\x00')

    def gen_ngrams(self, text):
        """
         @brief Generate ngrams from text. This is a convenience method for ` nltk. ngrams `
         @param text text to generate ngrams from
         @return list of ngrams ( tuples of length N_
        """
        return ngrams(text.split(), self.NGRAM)
    
    def add_padding(self, words):
        """
         @brief Add padding to word list. This is used to pad words that are too long to fit in NGRAM
         @param words List of words to be padded
         @return Word list with padding at the end ( 0x00
        """
        lenght = len(words)
        num_pad = 0
        # Pad the number of bytes to the end of the buffer.
        if lenght < self.NGRAM:
            num_pad = self.NGRAM - lenght
        words = ['\x00']*num_pad + words
        return ' '.join(words)
    def add_accent(self, text):
        """
         @brief Given text guess and add accented words to the text. This is used for text that is part of a sentence or sentence consisting of multiple words.
         @param text text to be guessed and added accents
         @return text with accented words added to it if there is
        """
        words = text.split(' ')
        lenght = len(words)
        # Add padding to the text if lenght is less than the NRAM
        if lenght < self.NGRAM:
            text = self.add_padding(words)
        ngrams = list(self.gen_ngrams(text.lower()))
        guessed_ngrams = list(self.guess(ngram) for ngram in ngrams)
        # Returns a string of ngrams in the form of a string of length lenght.
        if len(guessed_ngrams) == 1:
            return ' '.join(guessed_ngrams[0].split(' ')[self.NGRAM -lenght:]).strip()
        candidates = [Counter() for _ in range(len(guessed_ngrams) + self.NGRAM - 1)]
        # Update candidates for each ngram in the guessed_ngrams.
        for nid, ngram in enumerate(guessed_ngrams):
            # Update candidates for each word in the string.
            for wid, word in enumerate(re.split(' +', ngram)):
                candidates[nid + wid].update([word])
        output = ' '.join(c.most_common(1)[0][0] for c in candidates)
        return output
    
    def need_restore(self, sentence, threshhold = 0.5):
        """
         @brief Check if we need to restore a sentence. This is based on how many words are in the sentence and how frequently they are in the sentence
         @param sentence the sentence to be checked
         @param threshhold the threshold for word restoration. Default is 0. 5
         @return whether or not we need to restore the sentence
        """
        # if len(sentence.split())<5:
        #     return False
        sentence_non_accent = self.remove_accent(sentence)
        # Returns true if the sentence is a non accent sentence.
        if sentence.strip() == sentence_non_accent.strip():
            return True
        count = 0
        lst_words =  sentence.split()
        lst_words_no_accent = sentence_non_accent.split()
        # Count the number of words in lst_words_no_accent.
        for index, word in enumerate(lst_words):
            # count of words in lst_words_no_accent.
            if word == lst_words_no_accent[index]:
                count+=1
        # Return true if the number of lst_words is less than threshhold.
        if count/len(lst_words) > threshhold:
            return True
        return False
# This function is called from the main module.
if __name__ == "__main__":
    # path_to_model_accent_restore.h5, chọn model có NGRAM phù hợp
    NGRAM = 5
    path_model = './pickle_folder/bilstm_5gram_fulldata.h5'
    accent_restore = AccentRestore(path_model, NGRAM = NGRAM)
    textes = ["xin chao", "xin", "chao","khoa hoc may", "khoa hoc may tinh","gioi thiệu khoa hoc may",\
        "gioi thieu ve khoa hoc may tinh", "toiiiiiii muonnnnnn mccsjnsjcsncnsncnjsncncsj ncs njsnsc",\
        "toiiiiiiiiiiiiiiiiiii muonnnnnnnnnnnnnnnnnnnnnnnnnn", "ai tạo ra bot"]
    # This function is used to generate the restoring of the textes.
    for text in textes:
        # This function is used to restore the accents.
        if accent_restore.need_restore(text):
            # print("Start restore accents...!!!")
            time_start = time.time()
            text = accent_restore.remove_accent(text)
            sentence_restored = accent_restore.add_accent(text)
            time_end = time.time()
            # print("Câu gốc:", text)
            # print("Câu đã thêm dấu:", sentence_restored)
            # print("Time inference: ", time_end - time_start)