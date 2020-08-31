"""Created on Fri Jul 31 08:38:52 2020.

@author: Veljko
"""
from model import *
from utils import load_data

video_urls = 'D:\\obama_dataset\\video_urls.txt'
mfcc_path = 'D:\\obama_dataset\\mfcc_new'
mouth_shapes_path = 'D:\\obama_dataset\\mouth_shapes\\480p\\haar'
validation = 0.2
fps = 30
save_dir = 'C:\\Projects\\Python projects\\DeepFake\\save\\obama_data'
X, Y = load_data(video_urls,
                 mfcc_path,
                 mouth_shapes_path,
                 validation,
                 fps,
                 save_dir,
                 reprocess=False,
                 norm_input=True,
                 norm_output=False)
rnn_net = BidirectionalSingleLayerLSTMmodel("SL_BLSTM_120units_d00", rnn_size=120, dropout=0.0)
rnn_net.load_data(X, Y)
rnn_net.train(n_epochs=20)
rnn_net1 = BidirectionalSingleLayerLSTMmodel("SL_BLSTM_150units_d00", rnn_size=150, dropout=0.0)
rnn_net1.load_data(X, Y)
rnn_net1.train(n_epochs=20)
rnn_net2 = BidirectionalSingleLayerLSTMmodel("SL_BLSTM_120units_d01", rnn_size=120, dropout=0.1)
rnn_net2.load_data(X, Y)
rnn_net2.train(n_epochs=20)
rnn_net3 = BidirectionalSingleLayerLSTMmodel("SL_BLSTM_150units_d01", rnn_size=150, dropout=0.1)
rnn_net3.load_data(X, Y)
rnn_net3.train(n_epochs=20)