import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTMCell, RNN

class SimpleBiLSTM(Model):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleBiLSTM, self).__init__(name='simple_bilstm')
        self.embed = Embedding(vocab_size, embed_size)

        #self.lstm = Bidirectional(LSTM(hidden_size))
        self.lstm1 = LSTMCell(hidden_size)
        self.lstm2 = LSTMCell(hidden_size)
        self.stacked_lstm = RNN([self.lstm1, self.lstm2], return_sequences=True)
        self.bilstm = Bidirectional(self.stacked_lstm)

        self.linear = Dense(5, activation='softmax')


    def call(self, inputs):
        embedded = self.embed(inputs)
        feature = self.bilstm(embedded)
        outputs = self.linear(feature)

        return outputs



