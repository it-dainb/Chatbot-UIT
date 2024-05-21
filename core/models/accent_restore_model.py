import os

import numpy as np
import tensorflow as tf
keras = tf.keras

import regex as re
import unidecode    
import itertools
import string
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from core.utils.logger import logger
from core.config.config import get_config
from core.utils.preprocessing import clean_text
import unicodedata

from nltk import ngrams as nltk_ngrams
from fast_map import fast_map

import os
import json
import joblib, pickle
from collections import Counter

LONGEST_LENGTH = 6
MAX_LINES = 2_000_000

class AccentRestoreModel:
    def __init__(self, path = None, database = None, verbose = True):
        """
         @brief Initialize instance. If path is None the configuration is loaded from config. ini
         @param path path to configuration. ini
         @param database database to use for model creation. Defaults to None
         @param verbose whether to print information about
        """
        # Load the configuration file and set the model to None
        if path is not None:
            self.load(path)
        else:
            self.config = {}
            self.model = None
            
        self.database = database
        self.accented_chars_vietnamese = [
            'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
            'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
            'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
            'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
            'í', 'ì', 'ỉ', 'ĩ', 'ị',
            'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
            'đ',
        ]
        self.accented_chars_vietnamese.extend([c.upper() for c in self.accented_chars_vietnamese])
        self.pad_token = '\x00'
        self.alphabet = list((f'{self.pad_token} _' + string.ascii_letters + string.digits + ''.join(self.accented_chars_vietnamese)))

        self.verbose = verbose

    def create_model(self, units=256) -> tf.keras.Model:
        """
         @brief Creates a Keras model to be used for training. This is an LSTM with Bidirectional Recurrent Dropout
         @param units Number of units in the LSTM
         @return Keras model ready to be used for training the
        """
        model = keras.Sequential()
        
        model.add(keras.layers.LSTM(units = units, input_shape=(self.config['max_length'], len(self.alphabet)), return_sequences=True))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = units, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(self.alphabet))))
        model.add(keras.layers.Activation('softmax'))

        return model

    def compile(

        self,
        lr: float,
        optimizer: tf.keras.optimizers = None, 
    ):
        """
         @brief Compile the model. This compiles the Keras model and returns a dictionary of metrics.
         @param lr learning rate to use for optimizing the model
         @param optimizer optimizer to use for
        """
        optimizer = optimizer if optimizer else keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ])

    def remove_accent(self, text):
        """
         @brief Remove accented characters from text. This is useful for translating text that is in a text field such as text_for_person and vice versa.
         @param text The text to unidecode. Must be unicode.
         @return The text with accented characters removed. If the text is not unicode it will be returned as is
        """
        return unidecode.unidecode(text)

    def extract_phrases(self, text):
        """
         @brief Extract phrases from text. This is used to extract phrases that are part of a word or phrase list.
         @param text text to extract phrases from. Must be a string
         @return list of strings each of which is a
        """
        pattern = r'\w[\w ]*|\s\W+|\W+'
        return re.findall(pattern, text)

    def gen_ngrams(self, text):
        """
         @brief Generate n - grams from text. This is a generator function that will iterate over the text and return a list of n - grams that are used for training the model
         @param text text to generate ngrams
        """
        
        words = text.split() + [self.pad_token] * max(0, self.config['ngram'] - len(text.split()))
        ngrams = nltk_ngrams(words, self.config['ngram'])
        
        # Yields a list of ngrams.
        for ngram in ngrams:
            yield ' '.join(ngram)

    def process_phrase(self, p):
        """
         @brief Process a phrase and return ngrams. N - grams are generated by : meth : ` gen_ngrams ` and the length of the ngram is less than
         @param p phrase to be processed.
         @return list of n - grams that form the phrase
        """
        ngrams = []

        # Add ngrams to the list of ngrams if max_length is not less than max_length.
        for ngr in self.gen_ngrams(p):
            # Add a new ngram to the list of ngrams if the max length is reached.
            if len(ngr) < self.config['max_length']:
                ngrams.append(ngr)
        
        return ngrams

    def clean_text(self, text):
        """
         @brief Cleans text according to the synonyms dictionary. This is a wrapper around clean_text that does not use the tokenizer.
         @param text The text to clean. Must be a string.
         @return A string with all non - alphabetic characters removed
        """
        return clean_text(text, self.database.synonyms_dictionary, tokenizer=False)

    def create_data(self, path, database = None):
        """
        @brief Creates data to be used in training. This is a method that takes a path to a text file and a database object as an argument.
        @param path The path to the text file
        @param database The database object to
        """

        ngrams_path = os.path.join(get_config("Path", "data"), f"{self.config['ngram']}_ngrams.pkl")  
        
        # Returns a list of ngrams.
        # Returns a list of words that are not in the clean text.
        if os.path.exists(ngrams_path):
            list_ngrams = joblib.load(ngrams_path)
        else:
            with open(path, "r", encoding="utf8") as f_r:
                lines = f_r.read().split("\n")

            # The number of lines in the database.
            # The number of lines in the database.
            if database is not None:
                total = len(lines)
                lines.extend(database.create_train_label_data()['x'])
                lines.extend(database.answer['Answer'].tolist())

                total = len(lines) - total
            
            phrases = itertools.chain.from_iterable(self.extract_phrases(text) for text in lines)
            phrases = [p.strip() for p in phrases]
            phrases = list(set(phrases))            
            phrases = phrases[:MAX_LINES] + phrases[-1 * total - 100:]

            del lines
            
            clean_phrases = []

            # Remove phrases that are not in the clean text.
            # Add phrases to clean_phrases list if it is too long.
            for phrase in tqdm(fast_map(self.clean_text, phrases, threads_limit = 20), total=len(phrases)):
                # Add phrase to clean_phrases list if it is too long.
                # Add phrase to clean_phrases list if it is too long
                if len(phrase.split()) >= self.config['ngram']:
                    clean_phrases.append(phrase)

            clean_phrases = list(set(clean_phrases))

            del phrases
            
            list_ngrams = []
            # Returns a list of ngrams for each phrase in the clean phrases.
            for ngrams in tqdm(fast_map(self.process_phrase, clean_phrases, threads_limit = 20), total=len(clean_phrases)):
                list_ngrams.extend(ngrams)

            list_ngrams = list(set(list_ngrams))
            joblib.dump(list_ngrams, ngrams_path, protocol=pickle.HIGHEST_PROTOCOL)
            
        train, test = train_test_split(list_ngrams, test_size=0.2, random_state=42)
        del list_ngrams
        return train, test


    def padding(self, text):
        """
         @brief Pad text to max_length. This is used to ensure that the text doesn't exceed self. config ['max_length']
         @param text The text to pad.
         @return The padded text with padding token and length set to self. config
        """
        text = self.pad_token + text + self.pad_token * max(0, self.config['max_length'] - len(text) - 1)
        return text

    def encode(self, text):
        """
         @brief Encodes text into a 1 - D array. The array is padded with NFC characters to make it suitable for use as a Brainfuck word embeddings.
         @param text The text to encode. Must be unicode.
         @return A 1 - D array of length self. config ['max_length '
        """
        text = unicodedata.normalize("NFC", text)
        text = self.padding(text)

        x = np.zeros((self.config['max_length'], len(self.alphabet)))

        # x i c. length x. shape. length
        for i, c in enumerate(text[:self.config['max_length']]):
            x[i, self.alphabet.index(c)] = 1
        
        return x

    def decode(self, x):
        """
         @brief Decode a sequence of character indices. This is a convenience method for use with
         @param x A list of character indices
         @return A string of decoded character indices in the alphabet of
        """
        x = x.argmax(axis=-1)
        
        return ''.join(self.alphabet[i] for i in x if i != 0)

    def generate_data(self, data, batch_size=128):
        """
         @brief Generate data for training. This is a generator that yields batches of data in one batch.
         @param data The data to be encoded. Each element is a string that can be used as input to the model.
         @param batch_size The size of the batch. Default is 128
        """
        cur_index = 0

        # Yields x y arrays of data.
        while True:
            x, y = [], []
            # Encode the data to the output stream.
            for _ in range(batch_size):  
                y.append(self.encode(data[cur_index]))
                x.append(self.encode(self.remove_accent(data[cur_index])))
                cur_index += 1
                
                # Move cursor to the beginning of the data set.
                if cur_index > len(data)-1:
                    cur_index = 0
            
            yield np.array(x), np.array(y)

    def train(self, epochs = 10, save_path= "Models", name = "accent", data_path = "Data/train_accent.txt", database = None, ngram = 5, batch_size=1024, lr = 0.001):
        """
         @brief Train the model. This is a wrapper for generating and training data and saving them to Keras.
         @param epochs Number of epochs to train the model. Default 10
         @param save_path Path to save the model and checkpoint. Default'Models '
         @param name Name of the model. Default'accent '
         @param data_path Path to the data directory. Default'Data / train_accent. txt '
         @param database Database to use for training. Default None.
         @param ngram Ngram to use for training. Default 5
         @param batch_size Batch size in each epoch. Default 1024
         @param lr Likelihood to use for training. Default 0.
        """
        save_path = os.path.join(save_path, name)
        os.makedirs(save_path, exist_ok=True)

        self.config['max_length'] = LONGEST_LENGTH * ngram
        self.config['ngram'] = ngram
        self.config['name'] = name
        
        train, test = self.create_data(data_path, database)

        train_generator = self.generate_data(train, batch_size=batch_size)
        test_generator = self.generate_data(test, batch_size=batch_size)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, 'checkpoint_model.keras'),
            save_best_only=True, 
            verbose=1,
            monitor='categorical_accuracy',
            mode='max',
        )
        
        es_callback = keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=2, mode='max', restore_best_weights=True)

        json.dump(
            self.config, 
            open(os.path.join(save_path, "config.json"), "w", encoding="utf-8"), 
            ensure_ascii=False,
            indent=4,
            sort_keys=True
        )


        self.model = self.create_model()
        self.compile(lr=lr)

        self.model.fit(
            train_generator, 
            epochs=epochs,
            steps_per_epoch=len(train)//batch_size, 
            callbacks=[checkpoint_callback, es_callback],
        )

        self.save(save_path)

        score = self.model.evaluate(test_generator, steps=len(test)//batch_size)
        logger.info(f"Score: {score}")

    def save(self, save_path):
        """
         @brief Saves the Keras model to the given path. This will create the directory if it doesn't exist.
         @param save_path Path to save the Keras model
        """
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save(os.path.join(save_path, "model.keras"))
        
        json.dump(
            self.config, 
            open(os.path.join(save_path, "config.json"), "w", encoding="utf-8"), 
            ensure_ascii=False,
            indent=4,
            sort_keys=True
        )

    def load(self, path):
        """
         @brief Load model from path. This is called by : meth : ` ~keras. model. Trainer. train `
         @param path Path to the model
        """
        # Check if path exists and is not a valid model.
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} not found! Please train model first!")
        
        self.config = json.load(open(os.path.join(path, "config.json"), "r", encoding="utf8"))

        print(os.path.isfile(os.path.join(path, "model.keras")))
        
        logger.info(f"Loading {self.config['name']} model from: {path}")
        self.model = tf.keras.models.load_model(os.path.join(path, "model.keras"))

    def replace_digits(self, sent_a, sent_b):
        """
         @brief Replaces digits in a sentence with another. This is used to make comparisons between sentences that look like " A " and " B "
         @param sent_a The sentence to be replaced. It is assumed that the digits are in lower case.
         @param sent_b The sentence to be replaced. It is assumed that the digits are in upper case.
         @return A sentence with digits replaced with other digits in the same way
        """
        replaced_sentence = ""
        # Replace all characters in sent_a and sent_b with their replaced sentence.
        for char_a, char_b in zip(sent_a, sent_b):
            # Replace all characters in the sentence with the character b.
            if char_b.isdigit():
                replaced_sentence += char_a
            else:
                replaced_sentence += char_b
        return replaced_sentence

    def predict(self, query):
        """
         @brief Predict the most frequent word for a query. This is a convenience method for use in testing.
         @param query The query to predict. Must be a string of length n_sentence or n_sentence_words.
         @return The most frequent word ( s ) for the query
        """
        ngrams = self.process_phrase(query)
        X = np.array([self.encode(self.remove_accent(ngram)) for ngram in ngrams])

        with tf.device('/cpu:0'):
            preds = self.model.predict(X, verbose = self.verbose)
            
        preds = [self.decode(pred) for pred in preds]

        candidates = [Counter() for _ in range(len(preds) + min(self.config["ngram"], len(preds[0].split())) - 1)]
        # Update candidates for each word in the list of candidates.
        for nid, ngram in enumerate(preds):
            # Update candidates for each word in the ngram.
            for wid, word in enumerate(re.split(' +', ngram)):
                idx = nid + wid
                # Update the candidates for the word in the list.
                if idx < len(candidates):
                    candidates[nid + wid].update([word])
                
        output = ' '.join(c.most_common(1)[0][0] for c in candidates)
        return self.replace_digits(query, output)

    def need_restore(self, sentence, threshold = 0.5):
        """
         @brief Check if we need to restore the sentence. This is based on how many words are part of the sentence and how many times they are used to count the number of words
         @param sentence The sentence to be analysed
         @param threshold The threshold at which to consider restoration.
         @return True if sentence is part of the sentence False otherwise
        """
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
        # Return true if the threshold is less than threshold.
        if count/len(lst_words) >= threshold:
            return True
        return False