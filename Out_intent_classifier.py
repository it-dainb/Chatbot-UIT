import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.sequence import *
from sklearn.model_selection import *
from preprocessing import *
from read_database import  *
import numpy as np
import time
from gensim.models import *
import log

def read_data(synonyms_dictionary):
    X_out = read_data_out_of_domain(synonyms_dictionary)
    X_in = read_data_in_domain(synonyms_dictionary)
    y_in = [1]*len(X_in)
    y_out = [0] * len(X_out)
    print(len(X_out), len(y_out))
    print(len(X_in), len(y_in))
    return X_in, y_in, X_out, y_out


def train_word2vec(sentences):
    w2v_model = Word2Vec(
        sentences,
        window=3,
        min_count=1,
        sg= 1,
        workers=4,
        seed = 42)
    return w2v_model

def train_model(X_data_vector,y_data_encode,MAX_SEQUENCE_LENGTH,word_index,EMBEDDING_DIM,word2vec_model):
    X_train, X_test, y_train, y_test = train_test_split(X_data_vector, y_data_encode, stratify=y_data_encode,
                                                        test_size=0.2, random_state=42)

    count = 0
    embedding_matrix = np.asarray([np.random.uniform(-0.01,0.01,EMBEDDING_DIM) for _ in range((len(word_index) + 1))])
    for word, i in word_index.items():
        try:
            embedding_vector = word2vec_model.wv.get_vector(word, norm=True)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                count += 1
        except:
            pass
    print("Number word in pretrained word2vec: ", count)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    he_normal_initializer = tf.keras.initializers.he_normal(seed=42)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1 = Conv1D(64, 2, kernel_initializer=he_normal_initializer, padding="same", activation="relu")(embedded_sequences)
    max_pool1 = GlobalMaxPooling1D()(l_cov1)
    ave_pool1 = GlobalAveragePooling1D()(l_cov1)
    feature_conv1 = Concatenate(axis=1)([max_pool1, ave_pool1])

    l_cov2 = Conv1D(64, 3, kernel_initializer=he_normal_initializer, padding="same", activation="relu")(embedded_sequences)
    max_pool2 = GlobalMaxPooling1D()(l_cov2)
    ave_pool2 = GlobalAveragePooling1D()(l_cov2)
    feature_conv2 = Concatenate(axis=1)([max_pool2, ave_pool2])

    l_cov3 = Conv1D(64, 4, kernel_initializer=he_normal_initializer, padding="same", activation="relu")(embedded_sequences)
    max_pool3 = GlobalMaxPooling1D()(l_cov3)
    ave_pool3 = GlobalAveragePooling1D()(l_cov3)
    feature_conv3 = Concatenate(axis=1)([max_pool3, ave_pool3])

    feature_sent = Concatenate(axis=1)([feature_conv1, feature_conv2, feature_conv3])

    fully_connected_1 = Dropout(0.25)(Dense(256, activation='relu', kernel_initializer=he_normal_initializer,kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(feature_sent))
    fully_connected_2 = Dropout(0.25)(Dense(128, activation='relu', kernel_initializer=he_normal_initializer,kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(fully_connected_1))
    outputs = Dense(1, activation='sigmoid')(fully_connected_2)

    model = Model(sequence_input, outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    checkpoint = ModelCheckpoint("best_model.hdf5", monitor="val_accuracy", save_best_only=True, mode="auto", period=1)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, mode='max',restore_best_weights=True)

    hist = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=32, shuffle=True, callbacks=[es_callback])
    score = model.evaluate(X_test, y_test, batch_size=32)
    print("Accuracy on the test set: ", score)
    # print(model.summary())
    return model
def predict_sent(sent, model, tokenizer,max_length, synonyms_dictionary):
    
    # sent = replace_synonym(sent, synonyms_dictionary)
    
    # print("Sent predict: ", sent)

    X_data_vector = tokenizer.texts_to_sequences([sent])
    Xtrain = pad_sequences(X_data_vector, maxlen=max_length, padding='post')
    preds = model.predict(Xtrain, verbose=None)
    if preds[0] > 0.6:
        log.debug("in domain")
        return 1
    else:
        log.debug("out domain")
        return 0

if __name__ == '__main__':

    synonyms_dictionary = read_share_knowledge(path_share="Data/Data.xlsx")


    X_in, y_in, X_out, y_out = read_data(synonyms_dictionary)
    
    X_train = X_in + X_out
    y_train = y_in + y_out
    
    sentences = [row.strip().split() for row in X_train]
    model_word2vec  = train_word2vec(sentences)
    tokenizer = Tokenizer(filters="", lower=True, split=' ', oov_token="UNK")
    tokenizer.fit_on_texts(X_train)
    X_data_vector = tokenizer.texts_to_sequences(X_train)
    max_length = max([len(s.split()) for s in X_train])
    Xtrain = pad_sequences(X_data_vector, maxlen=max_length, padding='post')
   
    
    #train model 
    model = train_model(Xtrain,np.array(y_train),max_length,tokenizer.word_index,50, model_word2vec)
    
    #save model
    model.save("pickle_folder/Out_intent_model.h5")
    
    with open('pickle_folder/Out_vectorizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('pickle_folder/Out_max_length.pickle', 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    while(True):
        query = input("Nhập truy vấn: ")
        query = processing(query,synonyms_dictionary)
        print("Query clearn: ", query)
        result = predict_sent(query, model, tokenizer,max_length)
        print(query)
        print(result)
        if result == 1:
            print("in domain")
        else:
            print("Out domain")
        print()
        ## 1 là thuộc domain, 0 là không thuộc domain
    
    
    