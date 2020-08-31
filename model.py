"""Created on Sun Apr  5 19:34:17 2020

@author: Veljko
(basic script info).
"""
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
'''
from utils import RNNbase
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.regularizers import l2
from tensorflow import float32 as tf_float32
from tensorflow.math import exp as tf_exp
from tensorflow.debugging import set_log_device_placement
from tensorflow.config.experimental import list_physical_devices, set_memory_growth, set_visible_devices


class SingleLayerLSTMmodel(RNNbase):
    def __init__(self, name, **kwargs):
        super(SingleLayerLSTMmodel, self).__init__(**kwargs)
        self.model_name = name
        self.rnn_size = kwargs.get("rnn_size", 60)


    def lstm_model(self, infer: bool = False):
        input_data = Input(shape=(self.sequence_len,
                                  self.input_dim),
                           dtype=tf_float32,
                           batch_size=None)
        LSTM_layer = LSTM(self.rnn_size,
                          return_sequences=True,
                          return_state=False,
                          stateful=False)
        lstm_output = LSTM_layer(input_data)
        #dropout_layer = Dropout(0.2)
        #dropout_output = dropout_layer(lstm_output)
        output_data = TimeDistributed(Dense(self.output_dim,activation='linear'))(lstm_output)
        model = Model(inputs=input_data,
                      outputs=output_data,
                      name=self.model_name)
        optimizer = Adam(learning_rate=self.learning_rate,
                         clipnorm=self.gradient_clip)
        model.compile(optimizer,
                      loss='mse',
                      metrics=[MeanSquaredError()])
        model.summary()
        return model

class BidirectionalSingleLayerLSTMmodel(RNNbase):
    def __init__(self, name, **kwargs):
        super(BidirectionalSingleLayerLSTMmodel, self).__init__(**kwargs)
        self.model_name = name
        self.rnn_size = kwargs.get("rnn_size", 60)
        self.dropout = kwargs.get("dropout", 0.2)

    def lstm_model(self, infer: bool = False):
        input_data = Input(shape=(self.sequence_len,
                                  self.input_dim),
                           dtype=tf_float32,
                           batch_size=None)
        BiLSTM_layer = Bidirectional(LSTM(self.rnn_size,
                          return_sequences=True,
                          return_state=False,
                          stateful=False,
                          dropout=self.dropout))
        lstm_output = BiLSTM_layer(input_data)
        '''
        dense = Dense(self.output_dim,
                      activation='linear')
        '''
        output_data = TimeDistributed(Dense(self.output_dim,activation='linear'))(lstm_output)
        model = Model(inputs=input_data,
                      outputs=output_data,
                      name=self.model_name)
        optimizer = Adam(learning_rate=self.learning_rate,
                         clipnorm=self.gradient_clip)
        model.compile(optimizer,
                      loss='mse',
                      metrics=[MeanSquaredError()])
        model.summary()
        return model

'''
class EncoderDecoderLSTM(RNNbase):
    def __init__(self, name, **kwargs):
        super(EncoderDecoderLSTM, self).__init__(**kwargs)
        self.model_name = name
        self.rnn_size = kwargs.get("rnn_size", 60)
        self.sequence_len = kwargs.get("sequence_len", 100)
        self.input = kwargs.get("input", '')
        self.input2 = kwargs.get("input2", '')
        self.guy = kwargs.get("guy", "Obama2")
        self.training_dir = "obama_data"
        self.fps = 29.97
        self.load_data()
        self.audioinput = len(self.input2)
        if (self.audioinput):
            self.input = self.input2

    def lstm_model(self, infer: bool = False):
        input_data = Input(shape=(self.sequence_len,
                                  self.input_dim),
                           dtype=tf_float32,
                           batch_size=self.batch_size)
        LSTM_layer = LSTM(self.rnn_size,
                          return_sequences=True,
                          return_state=False,
                          stateful=False)
        lstm_output = LSTM_layer(input_data)
        output_data = TimeDistributed(Dense(self.output_dim,activation='linear'))(lstm_output)
        model = Model(inputs=input_data,
                      outputs=output_data,
                      name=self.model_name)
        optimizer = Adam(learning_rate=self.learning_rate,
                         clipnorm=self.gradient_clip)
        model.compile(optimizer,
                      loss='mse',
                      metrics=[MeanSquaredError()])
        model.summary()
        return model
'''
'''
a = BidirectionalRNNmodel(name='SimpleBLSTM_10batch',
                          n_epochs=300,
                          rnn_size=100,
                          batch_size =200,
                          decay_rate=1,
                          time_delay=0)
a.train()
'''
class MultiLayerLSTMmodel(RNNbase):
    def __init__(self, name, **kwargs):
        super(MultiLayerLSTMmodel, self).__init__(**kwargs)
        self.model_name = name
        self.rnn_size = kwargs.get("rnn_size", 60)
        self.num_of_layers = kwargs.get("num_of_layers", 2)


    def lstm_model(self, infer: bool = False):
        model = Sequential()
        model.add(Input(shape=(self.sequence_len,
                                  self.input_dim),
                           dtype=tf_float32,
                           batch_size=None))
        for i in range(self.num_of_layers):
            model.add(LSTM(self.rnn_size,
                          return_sequences=True,
                          return_state=False,
                          stateful=False))
        model.add(TimeDistributed(Dense(self.output_dim,activation='linear')))
        optimizer = Adam(learning_rate=self.learning_rate,
                         clipnorm=self.gradient_clip)
        model.compile(optimizer,
                      loss='mse',
                      metrics=[MeanSquaredError()])
        model.summary()
        return model
    
class MultiLayerBidirectionalLSTMmodel(RNNbase):
    def __init__(self, name, **kwargs):
        super(MultiLayerBidirectionalLSTMmodel, self).__init__(**kwargs)
        self.model_name = name
        self.rnn_size = kwargs.get("rnn_size", 60)
        self.num_of_layers = kwargs.get("num_of_layers", 2)


    def lstm_model(self, infer: bool = False):
        model = Sequential()
        model.add(Input(shape=(self.sequence_len,
                                  self.input_dim),
                           dtype=tf_float32,
                           batch_size=None))
        for i in range(self.num_of_layers):
            model.add(Bidirectional(LSTM(self.rnn_size,
                          return_sequences=True,
                          return_state=False,
                          stateful=False,
                          dropout=0.2)))
        model.add(TimeDistributed(Dense(self.output_dim,activation='linear')))
        optimizer = Adam(learning_rate=self.learning_rate,
                         clipnorm=self.gradient_clip)
        model.compile(optimizer,
                      loss='mse',
                      metrics=[MeanSquaredError()])
        model.summary()
        return model