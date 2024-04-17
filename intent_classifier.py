import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import *
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.sequence import *
from sklearn.model_selection import *
from preprocessing import *
from read_database import  *
import numpy as np
import time
from gensim.models import *
import log

# LOAD CONFIGN FOR CHATBOT CIMB
with open("./config/chatbot_config.txt", "r",encoding='utf-8') as file:
    content_config = file.read().split("\n")

for line in content_config:
    if line.startswith("NUMBER_OF_RECOMMENDATION_RESULT"):
        NUMBER_OF_RECOMMENDATION_RESULT = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_CONFIDENCE_DEFAULT"):
        THRESHOLD_INTENT_CONFIDENCE_DEFAULT = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_CONFIDENCE_TRACKING"):
        THRESHOLD_INTENT_CONFIDENCE_TRACKING = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_CONFIDENCE_GREETING"):
        THRESHOLD_INTENT_CONFIDENCE_GREETING = float(line.split("=")[1])
    if line.startswith("THRESHOLD_ERROR_FOR_AGENT"):
        THRESHOLD_ERROR_FOR_AGENT = float(line.split("=")[1])
    if line.startswith("THRESHOLD_RETURN_CONFIDENCE"):
        THRESHOLD_RETURN_CONFIDENCE = float(line.split("=")[1])
    if line.startswith("THRESHOLD_INTENT_FOR_NOT_TRACKING"):
        THRESHOLD_INTENT_FOR_NOT_TRACKING = float(line.split("=")[1])

# Read dataset from paths
#THEM CODE DOT 4
def read_data(synonyms_dictionary, file_name="Data/Data.xlsx"):
    data = pd.read_excel(file_name, sheet_name="Question", skiprows=2)

    X = data["Question"].tolist()
    y = data["Pattern Template"].tolist()

    list_label = []
    for item in set(y):
        label = item.split("|")[0].strip()
        
        if label.strip() in ["None", "", "truy_vấn_ngoài_phạm_vi"]:
            continue

        if label.strip() in list_label:
            continue

        list_label.append(label)

    X_data = []
    y_data = []
    for index, item in enumerate(X):
        label = y[index].split("|")[0].strip()
        if label in list_label:
            X_data.append(processing_train(item,synonyms_dictionary))
            y_data.append(list_label.index(label))
    
    print("Số lượng class là: ", len(list_label))
    print(list_label)

    # Save model
    with open('pickle_folder/list_label.pickle', 'wb') as handle:
        pickle.dump(list_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return  X_data, y_data, len(list_label)


def get_max2index_value(prob_array):
    max_value = max(np.array(prob_array))
    max_index = list(prob_array).index(max_value)
    return max_value, max_index


# Get intent label from intent classifer
def intent_classification(user_query_clean, model, vectorizer, label_list,max_length=25):

    x_feature = np.array(pad_sequences(vectorizer.texts_to_sequences([user_query_clean]), maxlen=max_length,padding="post"))

    pred_prob = model.predict(x_feature, verbose=None)

    max_value, max_index = get_max2index_value(pred_prob[0])
    intent1 = label_list[max_index]

    new_arr = [np.where(a == max_value, 0, a) for a in pred_prob[0]]

    max_value2, max_index2 = get_max2index_value(new_arr)

    max_index2 = list(pred_prob[0]).index(max_value2)
    intent2 = label_list[max_index2]
    
    log.debug("intent_classification:", intent1)

    return intent1, max_value, max_index, intent2, max_value2, max_index2

def train_word2vec(sentences):
    w2v_model = Word2Vec(
        sentences,
        vector_size=50,
        window=3,
        min_count=1,
        sg= 1,
        workers=4,
        seed = 42,
        epochs =100)
    return w2v_model

def train_model(X_data_vector,y_data_encode,MAX_SEQUENCE_LENGTH,word_index,EMBEDDING_DIM,word2vec_model):
    X_train, X_test, y_train, y_test = train_test_split(X_data_vector, y_data_encode, stratify=y_data_encode,
                                                        test_size=0.1, random_state=42)

    #X_train = X_data_vector
    #y_train = y_data_encode
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
    outputs = Dense(number_classes, activation='softmax')(fully_connected_2)

    model = Model(sequence_input, outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    checkpoint = ModelCheckpoint("best_model.hdf5", monitor="val_accuracy", save_best_only=True, mode="auto", period=1)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, mode='max',restore_best_weights=True)

    hist = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=32, shuffle=True, callbacks=[es_callback])
    score = model.evaluate(X_test, y_test, batch_size=32)
    print("Accuracy on the test set: ", score)
    #print(model.summary())
    return model


if __name__ == '__main__':

    synonyms_dictionary = read_share_knowledge(path_share="Data/Data.xlsx")
    start = time.time()
    X_data, y_data, number_classes = read_data(synonyms_dictionary,file_name="Data/Data.xlsx")
    print(len(X_data), len(y_data))
    print("Number class: ", number_classes)


    #Train word2vec
    start2 = time.time()
    sentences = [row.strip().split(" ") for row in X_data]
    print(sentences[0])
    word2vec_model = train_word2vec(sentences)
    print(len(word2vec_model.wv))
    end2 = time.time()
    print("Time for Training w2v model: ", end2 - start2)

    # Vectorize data
    y_data_encode = tf.keras.utils.to_categorical(y_data,num_classes=number_classes)

    # create the tokenizer
    tokenizer = Tokenizer(filters="", lower=True, split=' ', oov_token="UNK")
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(X_data)

    X_data_vector = tokenizer.texts_to_sequences(X_data)
    max_length = max([len(s.split()) for s in X_data])
    print("maxLength: ", max_length)
    Xtrain = pad_sequences(X_data_vector, maxlen=max_length, padding='post')
    vocab_size = len(tokenizer.word_index) + 1

    print(tokenizer.word_index)
    model = train_model(Xtrain,y_data_encode,max_length,tokenizer.word_index,50, word2vec_model)
    end = time.time()
    print("Time: ", end - start)

    # Save model
    model.save("pickle_folder/intent_model.h5")
    # Save vectorize
    with open('pickle_folder/vectorizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('pickle_folder/max_length.pickle', 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)


    vectorizer = read_vectorize()
    label_list = read_list_label()
    intent_model = read_trained_model()
    while(True):
        query = input("Nhập truy vấn: ")
        user_query_clean = processing(query,synonyms_dictionary)
        print("Pre-processing: ", user_query_clean)
        start = time.time()
        intent1, max_value, max_index, intent2, second_max_value, second_max_index = intent_classification(user_query_clean,
                                                                                                           intent_model,
                                                                                                           vectorizer,
                                                                                                           label_list, max_length)
        print("intent2: ", intent1)
        print("confident intent: ", max_value)
        end = time.time()
        print("Time Inference: ", end - start)
