import numpy as np
import tensorflow as tf
keras = tf.keras

from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from enum import Enum

import joblib
import pickle

import os
import json
from core.utils.logger import logger

import tensorflow_addons as tfa

class LossType(Enum):
    binary: str = "binary_crossentropy"
    categorical: str = "categorical_crossentropy"

class ClsModel:
    def __init__(self, path: str = None, verbose = True):
        """
         @brief Initialize the instance. If path is None the instance is reset to its initial state. If path is a string it is treated as a path to a JSON file containing the configuration of the object
         @param path The path to the JSON file
         @param verbose Whether to print information about
        """
        # Load the file at path.
        if path is None:
            self.reset()
        else:
            self.load(path)

        self.verbose = verbose

    def create_feature_conv(
        self,
        filters: int,
        kernel_size: int,
        layer, 
        kernel_initializer: tf.keras.initializers.Initializer, 
    ):
        """
         @brief Creates a 1D convolutional layer. It is used to train and test Keras models. In this case we are going to use Conv2D and Concatenate to get feature_conv = l_conv + max_pool + ave_pool
         @param filters Number of filters in the convolutional layer
         @param kernel_size Size of the convolution kernel in HWC
         @param layer Layer to be used for the convolutional layer
         @param kernel_initializer Initializer for the convolutional layer
         @return Layer with 2 convolutional layers : max_pool and averaged_pool. Output is a tensor
        """
        l_conv = keras.layers.Conv1D(filters, kernel_size, kernel_initializer=kernel_initializer, padding="same", activation="relu")(layer)
        max_pool = keras.layers.GlobalMaxPooling1D()(l_conv)
        ave_pool = keras.layers.GlobalAveragePooling1D()(l_conv)
        feature_conv = keras.layers.Concatenate(axis=1)([max_pool, ave_pool])

        return feature_conv

    def create_fully_connect(
        self, 
        units: int,
        layer, 
        kernel_initializer: tf.keras.initializers.Initializer, 
        l2: float=0.001, 
        dropout: float=0.25
    ):
        """
         @brief Creates a fully connected layer. It is used to create an input and output of a convolutional neural network.
         @param units Number of neurons in the network. Must be greater than 1.
         @param layer Layer to be connected to. Must be a layer of type Conv2D or Conv2DWithHilbertSpace.
         @param kernel_initializer Initializer for the kernel. Must be a Keras initializer.
         @param l2 L2 regularizer for the layer.
         @param dropout Dropout factor for the layer. Must be a Keras initializer.
         @return A fully connected layer with the given parameters. Note that the layer will be created in the same way
        """
        dense = keras.layers.Dense(
            units, 
            activation='relu', 
            kernel_initializer=kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(l2=l2)
        )(layer)

        fully_connect = keras.layers.Dropout(dropout)(dense)
        return fully_connect
    
    def create_model(self) -> tf.keras.Model:
        """
         @brief Creates a Keras model. This is an implementation of the Model interface used to train the model.
         @return A Keras model ready to be fed into the model_fn argument of tf. keras. Estimator
        """
        embedding_layer = keras.layers.Embedding(
            self.config['vocab_size'],
            self.config['embedding_dim'], 
            input_length=self.config['max_length'], 
            trainable=True
        )
        embedding_layer.build((None, ))
        embedding_layer.set_weights([self.embedding_matrix])

        he_normal_initializer = keras.initializers.he_normal(seed=42)

        sequence_input = keras.layers.Input(shape=(self.config['max_length'],), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        features_convs = [
            self.create_feature_conv(64, i, embedded_sequences, he_normal_initializer)
            for i in range(2, 5)
        ]
        feature_sent = keras.layers.Concatenate(axis=1)(features_convs)

        fully_connected_1 = self.create_fully_connect(256, feature_sent, he_normal_initializer)
        fully_connected_2 = self.create_fully_connect(128, fully_connected_1, he_normal_initializer)
        
        outputs = keras.layers.Dense(self.config['num_class'], activation='softmax')(fully_connected_2)

        model = keras.Model(sequence_input, outputs)
        return model

    def train_word2vec(self, X_data):
        """
         @brief Train word2vec model with data. This is a method to be used in conjunction with train_model
         @param X_data List of sentences each of which is a list of word
         @return Word2Vec model for training the neural network with X_data as input and output as output
        """
        logger.info("Training word2vec model")
        
        sentences = [row.strip().split(" ") for row in X_data]
        w2v_model = Word2Vec(
            sentences,
            vector_size=50,
            window=3,
            min_count=1,
            sg= 1,
            workers=4,
            seed = 42,
            epochs =100
        )
        
        return w2v_model

    def train_tokenizer(self, X_data):
        """
         @brief Train a tokenizer on the data. This is a convenience method for use in testing. It will return a Keras tokenizer and vocab size
         @param X_data data to be used for training
         @return tokenizer and vocab size for the training data ( int ) or None if there is no tokenizer ( None
        """
        logger.info("Training tokenizer")
        
        tokenizer = keras.preprocessing.text.Tokenizer(filters="", lower=True, split=' ', oov_token="UNK")
        tokenizer.fit_on_texts(X_data)

        vocab_size = len(tokenizer.word_index) + 1

        return tokenizer, vocab_size
    
    def reset(self):
        """
         @brief Reset the state of the model. This is called when you want to re - use the model for a new
        """
        self.model = None
        self.tokenizer = None
        self.w2v_model = None
        self.config = {}

    def prepare_X(self, X_data):
        """
         @brief Prepare data for training. This is a wrapper around preprocessing to convert text to sequences and pad to word length
         @param X_data list of strings or single string
         @return X_data_padded list of sequences padded to word length or single string depending on config.
        """
        # If X_data is a string it is converted to a list of strings.
        if isinstance(X_data, str):
            X_data = [X_data]

        X_data_vector = self.tokenizer.texts_to_sequences(X_data)
        X_data_padded = keras.preprocessing.sequence.pad_sequences(X_data_vector, maxlen=self.config['max_length'], padding='post')

        return X_data_padded

    def prepare_y(self, y_data):
        """
         @brief Convert the data to Categorical. This is called before training to make it easier to use in inference
         @param y_data A tensor of shape [ batch_size num_class ]
         @return A tensor of shape [ batch_size num_class ] where each element is a 1 - D
        """
        return tf.keras.utils.to_categorical(y_data, num_classes=self.config['num_class'])
    
    def prepare_data(self, X_data, y_data):
        """
         @brief Prepare data for training and testing. This is a wrapper around the : func : ` ~gensim. models. train_test_split ` function to split the data into training and testing sets
         @param X_data Data to be used for training
         @param y_data Target values for training ( 0 - 1 )
         @return A tuple of : class : ` numpy. ndarray ` of shape ( n_samples n_features
        """
        
        X_data_encode = self.prepare_X(X_data)
        y_data_encode = self.prepare_y(y_data)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data_encode, y_data_encode, 
            stratify=y_data_encode,
            test_size=0.1, 
            random_state=42
        )

        return X_train, X_test, y_train, y_test

    def get_embedding_matrix(self):
        """
         @brief Generates a matrix of word embeddings. This is used to determine which words are part of the vocabulary and which are not in the vocabulary.
         @return An N x N matrix where N is the number of words in the vocabulary and each row is a word
        """
        embedding_matrix = np.asarray([np.random.uniform(-0.01, 0.01, self.config['embedding_dim']) for _ in range(self.config['vocab_size'])])
        # Returns the embedding vector for each word in the vocabulary.
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = self.word2vec_model.wv.get_vector(word, norm=True)
                # Set the embedding vector to the embedding vector.
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            except:
                pass

        return embedding_matrix

    def compile(
        self,
        lr: float,
        optimizer: tf.keras.optimizers = None, 
        loss: LossType = LossType.binary,
    ):
        """
         @brief Compile the model. This is a convenience method for compiling the model with Keras optimizers.
         @param lr learning rate to use for training and eval.
         @param optimizer optimizer to use for compilation. If None a Adam optimizer will be used.
         @param loss Loss type to use for loss computations
        """
        optimizer = optimizer if optimizer else keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss=loss.value, optimizer=optimizer, metrics=[
            keras.metrics.BinaryAccuracy() if loss == LossType.binary else keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ])
    
    def train(
        self, 
        data,
        embedding_dim: int = 50,
        name = "cls_model",
        lr: float = 0.001, 
        optimizer: tf.keras.optimizers = None, 
        loss: LossType = LossType.binary,
        reset: bool = False,
        save_path: str = "Models"
    ):       
        """
         @brief Train the Keras model. This is the main entry point for training the Keras model. You can call it yourself as an instance of the Keras class or in a subclass if you want to use different optimizers or loss functions.
         @param data Dictionary containing x and y. Each key should be a string and the value should be a numpy array of shape [ num_samples num_classes ]
         @param embedding_dim Dimension of embeddings to use.
         @param name Name of the Keras model. Default is " cls_model "
         @param lr Loss parameter for the Keras model.
         @param optimizer Optimizer to use for training ( default is None )
         @param loss Loss parameter for the Keras model.
         @param reset If True the model will be reset to initial state ( default is False )
         @param save_path Path to save the model to ( default is " Models "
        """ 
        X_data = data['x']
        y_data = data['y']

        logger.info(f"Training {name} model")
        logger.debug(f"Data Total: {len(X_data)}")

        self.config['embedding_dim'] = embedding_dim
        self.config['num_class'] = data['num_class']
        self.config['max_length'] = data['max_length']
        self.config['class'] = data['class']
        self.config['name'] = name

        # Reset the state of the object to its initial state.
        if reset:
            self.reset()

        self.w2v_model = self.train_word2vec(X_data)
        self.tokenizer, vocab_size = self.train_tokenizer(X_data)

        self.config['vocab_size'] = vocab_size
        
        self.embedding_matrix = self.get_embedding_matrix()
        
        # Create a new model if it doesn t exist.
        if self.model is None:
            self.model = self.create_model()

        logger.info("Compile model")
        self.compile(lr, optimizer, loss)

        es_callback = keras.callbacks.EarlyStopping(monitor='val_binary_accracy' if loss == LossType.binary else 'val_categorical_accuracy', patience=15, mode='max', restore_best_weights=True)

        X_train, X_test, y_train, y_test = self.prepare_data(X_data, y_data)

        logger.debug(f"Train : {len(X_data)}")
        logger.debug(f"Test  : {len(X_test)}")

        logger.info("Training model")
        hist = self.model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=32, shuffle=True, callbacks=[es_callback])

        logger.info("Evaluate model")
        score = self.model.evaluate(X_test, y_test, batch_size=32)

        logger.info(f"Score: {score}")

        self.save(name, save_path)

    def save(self, name, path):
        """
         @brief Save the Keras model to disk. This will be called by : meth : ` keras. models. BaseModel. save `
         @param name The name of the saved model.
         @param path The directory to save the model to. It will be created if it doesn't exist
        """
        save_path = os.path.join(path, name)

        os.makedirs(save_path, exist_ok=True)
            
        self.model.save(os.path.join(save_path, "model.keras"))

        joblib.dump(self.tokenizer, os.path.join(save_path, "tokenizer.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(self.w2v_model, os.path.join(save_path, "w2v_model.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(self.embedding_matrix, os.path.join(save_path, "embedding_matrix.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
        
        json.dump(
            self.config, 
            open(os.path.join(save_path, "config.json"), "w", encoding="utf-8"), 
            ensure_ascii=False,
            indent=4,
            sort_keys=True
        )

    def load(self, path):
        """
         @brief Load model from path. This is called by : meth : ` ~gensim. models. tfa. Trainer. train `
         @param path Path to the model
        """
        # Check if path exists and is not a valid model.
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} not found! Please train model first!")
        
        self.config = json.load(open(os.path.join(path, "config.json"), "r", encoding="utf8"))
        logger.info(f"Loading {self.config['name']} model from: {path}")
        
        self.model = tf.keras.models.load_model(os.path.join(path, "model.keras"), custom_objects={"Addons>F1Score": tfa.metrics.F1Score})
        self.tokenizer = joblib.load(os.path.join(path, "tokenizer.pkl"))
        self.w2v_model = joblib.load(os.path.join(path, "w2v_model.pkl"))
        self.embedding_matrix = joblib.load(os.path.join(path, "embedding_matrix.pkl"))
    
    async def predict(self, query):
        """
         @brief Predict class probabilities for a query. This is a coroutine. You can call it from any thread.
         @param query Query to predict class probabilities for. It should be a tensor of shape [ batch_size num_query_words ]
         @return List of class probabilities
        """
        x_feature = self.prepare_X(query)

        with tf.device('/cpu:0'):
            pred_prob = self.model.predict(x_feature, verbose=self.verbose)[0]
            
        class_prob = [
            {
                'class': self.config['class'][idx],
                'prob': prob
            }
            for idx, prob in enumerate(pred_prob)
        ]

        class_prob = sorted(class_prob, key = lambda x: x['prob'], reverse=True)

        return class_prob