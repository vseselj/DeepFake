"""Created on Sun Apr  5 15:17:28 2020.

@author: Veljko
(basic script info).
"""
import os
import numpy as np
from typing import Tuple, List
import pickle
from math import floor, ceil
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.backend import get_value, set_value
import bisect
import random
import matplotlib.pyplot as plt
import pandas as pd


def get_input_vector(audio,
                     start_frame: int,
                     n_frames: int,
                     fps: float) -> Tuple[int, int, np.ndarray]:
    """Create feature vector from loaded cepstral coefficients.

    Based on start frame and number of frames in video, subset of audio
    signal is taken. Function also concatenates temporal derivatives of
    cepstral coefficients.

    Args:
        audio (ndarray): 15 column vector that contains 13 cepstral
            ceofficients, log mean energy and timestamps.
        audio_diff (ndarray): temporal derivative of cepstral coeficients
            and log mean energy.
        timestamps (ndarray): vector that contains timestamp of each frame.
        start_frame (int): number of starting frame.
        n_frames (int): number of frames.

    Returns:
        Tuple[int, int, np.ndarray]: Start and end index that corresponds to
            start and end frame and fetaure vector that contains 28
            columns, 13 cepstral coefficients and log mean energy followed
            by their temporal derivatives.

    """
    start_audio = bisect.bisect_left(audio['Timestamps'],
                                     (start_frame - 1)/fps)
    end_audio = bisect.bisect_right(audio['Timestamps'],
                                    (start_frame + n_frames - 2)/fps)
    features = audio[audio.columns[1:]][start_audio:end_audio]
    return start_audio, end_audio, features.to_numpy(dtype=np.float32)

def get_output_vector(mouth_shape,
                      audio_timestamps,
                      start_audio,
                      end_audio,
                      start_frame,
                      fps):
    Y = np.zeros((end_audio-start_audio, mouth_shape.shape[1]),
                 dtype=np.float32)
    j = 0
    for audio_ind in range(start_audio, end_audio):
        timestamp = audio_timestamps[audio_ind]
        while timestamp >= (start_frame-1+j+1)/fps:
            j += 1
        t = (timestamp-(start_frame-1+j)/fps)*fps
        Y[audio_ind - start_audio, :] = mouth_shape[j, :]*(1-t)\
            + mouth_shape[min(len(mouth_shape)-1, j+1), :]*t
    return Y

def load_data(video_urls:str,
              mfcc_path: str,
              mouth_shapes_path: str,
              validation: float,
              fps: float,
              save_dir,
              reprocess=False,
              norm_input=True,
              norm_output=False):
    if reprocess:
        youtube_ids = []
        inputs = {'training': [], 'validation': []}
        outputs = {'training': [], 'validation': []}
        with open(video_urls, 'r') as f:
            for line in f.readlines():
                youtube_ids.append(line.split('=')[1].replace('\n', ''))
        for youtube_id in youtube_ids:
            audio = pd.read_csv(mfcc_path+os.sep+youtube_id+'.csv')
            video_parts = os.listdir(mouth_shapes_path+os.sep+youtube_id)
            for video_part in video_parts:
                print('Processing video %s part %s' % (youtube_id, video_part))
                if random.random() > validation:
                    set_name = 'training'
                else:
                    set_name = 'validation'
                skip = os.path.isfile(mouth_shapes_path + os.sep +
                                      youtube_id + os.sep +
                                      video_part + os.sep +
                                      'report.txt')
                if not(skip):
                    mouth_shape = pd.read_csv(mouth_shapes_path + os.sep +
                                              youtube_id + os.sep +
                                              video_part + os.sep +
                                              'mouth_shapes.csv',
                                              header=None)
                    start_frame = mouth_shape[0][0]
                    n_frames = len(mouth_shape[0])
                    start_audio, end_audio, X = get_input_vector(audio,
                                                                 start_frame,
                                                                 n_frames,
                                                                 fps)
                    Y = get_output_vector(mouth_shape[mouth_shape.columns[1:]].to_numpy(),
                                          audio['Timestamps'].to_numpy(),
                                          start_audio,
                                          end_audio,
                                          start_frame,
                                          fps)
                    inputs[set_name].append(X)
                    outputs[set_name].append(Y)
        (mean_inputs, std_inputs,
         mean_outputs, std_outputs) = normalize(inputs,
                                                outputs,
                                                save_dir,
                                                norm_input=norm_input,
                                                norm_output=norm_output)
        f = open(save_dir+os.sep+"obama_data.cpkl", 'wb')
        pickle.dump({"input": inputs["training"],
                     "input_mean": mean_inputs,
                     "input_std": std_inputs,
                     "output": outputs["training"],
                     "output_mean": mean_outputs,
                     "output_std": std_outputs,
                     "val_input": inputs["validation"],
                     "val_output": outputs["validation"]}, f, protocol=2)
        f.close()
    else:
        f = open(save_dir+os.sep+"obama_data.cpkl", "rb")
        data = pickle.load(f)
        inputs = {"training": data["input"],
                  "validation": data["val_input"]}
        outputs = {"training": data["output"],
                   "validation": data["val_output"]}
        f.close()
    return inputs, outputs

def normalize_data(data: List[List[float]],
                   save_directory: str,
                   name: str,
                   var_names: List[str],
                   normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean value and standard devation and normalize the data.

    Mean value and standard deviation is calculated for vector of all data, and
    for each feture (column-wise). Minimum and maximum value for each feature
    is also caluclated. Calcluated values are saved in .npy and .txt files.

    Args:
        data (List[List[float]]): Data from all files saved in each list.
        save_directory (str): Directory where calculated data is saved.
        name (str): Name of the save file.
        var_names (List[str]): List of fetaure names.
        normalize (bool, optional): If True data will be normalized. If False,
            mean and standard deviation will have values 0 and 1.Defaults to
            True.

    Returns:
        Tuple[ndarray, ndarray]: Funtion returns calculated mean value and
            standard deviation.

    """
    all_data = np.concatenate(data)
    means = np.mean(all_data, axis=0)
    sigmas = np.std(all_data, axis=0)
    f = open(save_directory + "\\" + name + ".txt", 'w')
    min_values = np.min(all_data, axis=0)
    max_values = np.max(all_data, axis=0)
    if not(isinstance(normalize, list)):
        normalize = [normalize]*len(var_names)
    for i, n in enumerate(var_names):
        if normalize[i]:
            f.write(n +
                    "\n  mean:%f\
                     \n  std :%f\
                     \n  min :%f\
                     \n  max :%f\n\n" %
                    (means[i], sigmas[i], min_values[i], max_values[i]))
        else:
            f.write(n +
                    "\n  mean:%f (-> 0)\
                     \n  std :%f (-> 1)\
                     \n  min :%f\
                     \n  max :%f\n\n" %
                    (means[i], sigmas[i], min_values[i], max_values[i]))
            means[i] = 0
            sigmas[i] = 1
    np.save(save_directory+os.sep+name+".npy",
            {'min': min_values,
             'max': max_values,
             'mean': means,
             'std': sigmas})
    f.close()
    for i in range(len(data)):
        data[i] = (data[i]-means)/sigmas
    return means, sigmas


def normalize(inputs,
              outputs,
              save_dir,
              norm_input=True,
              norm_output=False):
    input_var_names = []
    output_var_names = []
    for i in range(inputs['training'][0].shape[1]):
        input_var_names.append("fea%02d" % i)
    for i in range(outputs['training'][0].shape[1]):
        output_var_names.append("fea%02d" % i)
    input_means, input_stds = normalize_data(inputs['training'],
                                             save_dir,
                                             "statinput",
                                             input_var_names,
                                             normalize=norm_input)
    output_means, output_stds = normalize_data(outputs['training'],
                                               save_dir,
                                               "statoutput",
                                               output_var_names,
                                               norm_output)
    for i in range(len(inputs['validation'])):
        inputs['validation'][i] = (inputs['validation'][i] -
                                   input_means)/input_stds
    for i in range(len(outputs['validation'])):
        outputs['validation'][i] = (outputs['validation'][i] -
                                    output_means)/output_stds
    return input_means, input_stds, output_means, output_stds


def delay_data(inputs: dict,
               outputs: dict,
               time_delay: int,
               sequence_len: float) -> Tuple[dict, dict]:
    """Delays input data for specified time delay.

    Args:
        inputs (dict): Input data.
        outputs (dict): Output data.

    Returns:
        Tuple[dict, dict]: Delayed input and output data.

    """
    new_inputs = {"training": [], "validation": []}
    new_outputs = {"training": [], "validation": []}
    for key in new_inputs:  # for validation and training
        for i in range(len(inputs[key])):  # for each input data
            if len(inputs[key][i]) - time_delay >=\
               (sequence_len + 2):
                if time_delay > 0:
                    new_inputs[key].append(
                        inputs[key][i][time_delay:])
                    new_outputs[key].append(
                        outputs[key][i][:-time_delay])
                else:
                    new_inputs[key].append(inputs[key][i])
                    new_outputs[key].append(outputs[key][i])
    return new_inputs, new_outputs


class LearningRateSchedulerPerBatch(LearningRateScheduler):
    """Callback class to modify the default learning rate scheduler to operate each batch."""

    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0  # Global batch index (the regular batch argument refers to the batch index within the epoch)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
        self.count += 1


class DataGenerator(Sequence):
    def __init__(self,
                 X_data,
                 Y_data,
                 batch_size,
                 sequence_len,
                 input_dim,
                 output_dim):
        if len(X_data) != len(Y_data):
            raise("Input and output data must have the same length")
        #self.X_data = []
        #self.Y_data = []
        self.X_data = X_data
        self.Y_data = Y_data
        self.X_data_ind = []
        self.Y_data_ind = []
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        for i in range(len(X_data)):
            seq_start_ind = 0
            while seq_start_ind+self.sequence_len < len(X_data[i]):
                #self.X_data.append(np.copy(X_data[i][seq_start_ind:seq_start_ind+self.sequence_len, :]))
                #self.Y_data.append(np.copy(Y_data[i][seq_start_ind:seq_start_ind+self.sequence_len, :]))
                self.X_data_ind.append([i, seq_start_ind])
                self.Y_data_ind.append([i, seq_start_ind])
                seq_start_ind += 1
        ind_list = list(range(len(self.X_data_ind)))
        random.shuffle(ind_list)
        self.X_data_ind = [self.X_data_ind[i] for i in ind_list]
        self.Y_data_ind = [self.Y_data_ind[i] for i in ind_list]

    def __len__(self):
        return int(len(self.X_data_ind)//self.batch_size)

    def __getitem__(self, index):
        indecies = [ind for ind in range(index*self.batch_size,
                                         (index+1)*self.batch_size)]
        X = np.empty((self.batch_size, self.sequence_len, self.input_dim),
                     dtype=np.float32)
        Y = np.empty((self.batch_size, self.sequence_len, self.output_dim),
                     dtype=np.float32)
        for ind in indecies:
            i = indecies.index(ind)
            X[i, :, :] = np.copy(self.X_data[self.X_data_ind[ind][0]][self.X_data_ind[ind][1]:self.X_data_ind[ind][1]+self.sequence_len, :])
            #Y[i, :, :] = np.copy(self.Y_data[indecies[i]][:, :])
            Y[i, :, :] = np.copy(self.Y_data[self.Y_data_ind[ind][0]][self.Y_data_ind[ind][1]:self.Y_data_ind[ind][1]+self.sequence_len, :])
        return X, Y


class RNNbase:
    """

    """
    def __init__(self, **kwargs):
        """
        

        Returns:
            None.

        """
        self.save_every_n_epochs = kwargs.get('save_every_n_epochs', 10)
        self.gradient_clip = kwargs.get('gradient_clip', 10)
        self.learning_rate = kwargs.get('learning_rate', 1E-3)
        self.decay_rate = kwargs.get('decay_rate', 1)
        self.keep_probability = kwargs.get('keep_probability', 1)
        self.time_delay = kwargs.get("time_delay", 20)
        self.batch_size = kwargs.get("batch_size", 100)
        self.sequence_len = kwargs.get("sequence_len", 100)
        self.fps = 30

    def load_data(self, inputs, outputs):
        """
        

        Returns:
            None.

        """
        self.input_dim = inputs['training'][0].shape[1]
        self.output_dim = outputs['training'][0].shape[1]
        self.train_data_gen = DataGenerator(inputs["training"],
                                            outputs["training"],
                                            self.batch_size,
                                            self.sequence_len,
                                            self.input_dim,
                                            self.output_dim)
        self.val_data_gen = DataGenerator(inputs["validation"],
                                          outputs["validation"],
                                          self.batch_size,
                                          self.sequence_len,
                                          self.input_dim,
                                          self.output_dim)

    def lstm_model(self):
        """Define LSTM model.

        Derived classes must overide this method.

        Raises:
            NotImplementedError: DESCRIPTION.

        Returns:
            None.

        """
        raise NotImplementedError

    def train(self, n_epochs):
        """Train model.

        Train model for specified number of epochs and save model checkpoints
        and history log.

        Returns:
            None.

        """
        self.model = self.lstm_model()
        if not os.path.exists("save\\" + self.model.name + "_checkpoints"):
            os.mkdir("save\\" + self.model.name + "_checkpoints")

        checkpoint_path = "save\\" + self.model.name +\
                          "_checkpoints\\cp-{epoch:d}.hdf5"
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=False,
                                      save_freq='epoch')
        lr_decay_callback = LearningRateSchedulerPerBatch(
                    lambda step: (self.learning_rate * self.decay_rate ** (step/50000)))
        log_path = "save\\" + self.model.name + "_checkpoints\\history.csv"
        log_callback = CSVLogger(log_path, separator=',', append=True)
        history = self.model.fit(x=self.train_data_gen,
                                 epochs=n_epochs,
                                 validation_data=self.val_data_gen,
                                 callbacks=[cp_callback,
                                            log_callback,
                                            lr_decay_callback],
                                 max_queue_size=100,
                                 verbose=2)
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

    def continue_training(self,
                          start_epoch: int,
                          n_epochs: int,
                          **kwargs) -> None:
        """Continue training of an model from the checkpoint.

        Load model from the "hdf5" and continue the training.

        Args:
            start_epoch (int): Start eoch of the new training session.
            n_epochs (int): Number of epochs in new training session.

        Returns:
            None.

        """
        checkpoint_path = "save\\" + self.model_name +\
                          "_checkpoints\\cp-{epoch:d}.hdf5"
        loaded_model = load_model(checkpoint_path.format(epoch=start_epoch))
        print("Old learning rate: ", get_value(loaded_model.optimizer.lr))
        if 'learning_rate' in kwargs:
            print("New learning rate: ", kwargs.get('learning_rate'))
            set_value(loaded_model.optimizer.lr, kwargs.get('learning_rate'))
        loaded_model.summary()
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=False,
                                      save_freq='epoch')
        log_path = "save\\" + self.model_name + "_checkpoints\\history.csv"
        log_callback = CSVLogger(log_path, separator=',', append=True)
        loaded_model.fit(x=self.train_data_gen,
                         batch_size=self.batch_size,
                         epochs=n_epochs+start_epoch,
                         validation_data=self.val_data_gen,
                         callbacks=[cp_callback, log_callback],
                         initial_epoch=start_epoch)
        prev_history = pd.read_csv(log_path)
        plt.figure()
        plt.plot(prev_history['loss'])
        plt.plot(prev_history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

    def test(self, X):
        checkpoint_path = 'save'+os.sep+self.model_name+'_checkpoints'
        history_path = checkpoint_path+os.sep+'history.csv'
        history =pd.read_csv(history_path)
        best_epoch = history['val_loss'].idxmin()+1
        model_path = checkpoint_path+os.sep+'cp-'+str(best_epoch)+'.hdf5'
        model = load_model(model_path)
        model.summary()
        Y = model.predict(X, batch_size=1)
        return Y
        