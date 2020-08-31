"""Created on Tue Aug 18 08:04:13 2020.

@author: Veljko
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

history_path = 'C:\\Projects\\Python projects\\DeepFake\\save\\SL_BLSTM_30units_checkpoints\\history.csv'
history = pd.read_csv(history_path)
ax = history.plot(x='epoch', y='loss')
ax.set_xlabel('Epoha')
ax.set_ylabel('Greshka na trenirajucem skupu')
ax.set_xticks(history['epoch'].to_list())
ax.legend(['MSE'])
fig = ax.get_figure()
fig.savefig('C:\\Users\\Veljko\\Desktop\\chap 4'+os.sep+'SL_BLSTM_30units_d02_checkpoint_train.png')
ax = history.plot(x='epoch', y='val_loss')
ax.set_xlabel('Epoha')
ax.set_ylabel('Greshka na validacionom skupu')
ax.legend(['MSE'])
ax.set_xticks(history['epoch'].to_list())
fig = ax.get_figure()
fig.savefig('C:\\Users\\Veljko\\Desktop\\chap 4'+os.sep+'SL_BLSTM_30units_d02_checkpoint_validation.png')