# author Ran
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
from .preprocessing import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.backend import clear_session
import tensorflow as tf
import keras.backend as K
import os
class seq2seqmodel:
    def __init__(self):
        self.pp = preprocessing()
        clear_session()
        try:
            self.encoder, self.decoder,self.graph, self.sess = self._load_model()
        except (FileNotFoundError, OSError):
            self._train_model()
            self.encoder, self.decoder,self.graph, self.sess = self._load_model()

    def _load_model(self):
        latent_dim = np.power(2, 5)
        # define an input sequence and process it
        encoder_inputs = Input(shape=(None,))
        x = Embedding(self.pp.num_encoder_tokens, latent_dim)(encoder_inputs)
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(x)
        # we discard outpurs and keep two states as initial states of decoder model\
        encoder_states = [state_h, state_c]
        # set up decoder
        decoder_inputs = Input(shape=(None,))
        x = Embedding(self.pp.num_decoder_tokens, latent_dim)(decoder_inputs)
        # we set up our decoder to return full output sequence,
        # we don't use the return state in training model but in inference
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
        decoder_dense = Dense(self.pp.num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)
        # load model if exists
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        filepath = self.full_path("seq2seq_weights.h5")
        model.load_weights(filepath)
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        model._make_predict_function()
        graph = tf.get_default_graph()
        sess = K.get_session()
        # create encoder and decoder
        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(x, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model, graph, sess


    def _train_model(self):
        latent_dim = np.power(2, 5)
        epochs = 200
        batch_size = 64
        # define an input sequence and process it
        encoder_inputs = Input(shape=(None,))
        x = Embedding(self.pp.num_encoder_tokens, latent_dim)(encoder_inputs)
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(x)
        # we discard outpurs and keep two states as initial states of decoder model\
        encoder_states = [state_h, state_c]
        # set up decoder
        decoder_inputs = Input(shape=(None,))
        x = Embedding(self.pp.num_decoder_tokens, latent_dim)(decoder_inputs)
        # we set up our decoder to return full output sequence,
        # we don't use the return state in training model but in inference
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
        decoder_dense = Dense(self.pp.num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)
        # load model if exists
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        filepath = self.full_path("seq2seq_weights.h5")
        checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
        callbacks_list = [checkpoint]
        model.fit(
            [self.pp.encoder_input_data, self.pp.decoder_input_data],
            self.pp.decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks = callbacks_list)
        print("model weights saved to disk")

    def encode_input(self, input_text):
        input_text = self.pp.clean_text(input_text)
        # convert text into integers
        input_ints = []
        ints = []
        i = 0
        for word in input_text.split():
            if i > 20:
                break
            if word in self.pp.questions_vocab_to_int:
                ints.append(self.pp.questions_vocab_to_int[word])
            else:
                ints.append(self.pp.questions_vocab_to_int["<UNK>"])
            i += 1
        new_ints = list(np.repeat(self.pp.questions_vocab_to_int["<PAD>"], self.pp.max_length - i))
        new_ints.extend(ints)
        input_ints.append(new_ints)
        input_ints = np.array(input_ints)
        return input_ints

    def full_path(self,filename):
        directory = __file__.split('\\')
        directory = '/'.join(directory[:-1])
        return directory + "/" + filename
    # map input to output
    def decode_sequence(self, input_seq):
        encoded_input = self.encode_input(input_seq)
        with self.sess.as_default():
            with self.graph.as_default():
                states_value = self.encoder.predict(encoded_input)
        # generate target sequence of length 1
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.pp.questions_vocab_to_int["<GO>"]
        decoded_sentence = ""
        while True:
            with self.sess.as_default():
                with self.graph.as_default():
                    output_tokens, h, c = self.decoder.predict([target_seq] + states_value)
            # sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.pp.answers_int_to_vocab[sampled_token_index]
            if sampled_word == "<EOS>":
                break
            decoded_sentence += sampled_word
            decoded_sentence += " "
            if len(decoded_sentence.split()) > self.pp.max_length:
                break
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
        return decoded_sentence

if  __name__ == "__main__":
    model = seq2seqmodel()
    print(model.decode_sequence("hey"))
