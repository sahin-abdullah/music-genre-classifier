import os
import re
import numpy as np
import pandas as pd
import librosa as lbr
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

class AudioProcess():
    """A class to process audio (.wav) formatted files for given settings.
    This object is written specifically for GTZAN dataset which has a following directory 
    structure. We suggest to use similar data structure for different dataset

    root directory
        ├ data
    |        ├ blues
    |        ├ classical
    |        ├ country
    |        ├ disco
    |        ├ hiphop
    |        ├ jazz
    |        ├ metal
    |        ├ pop
    |        ├ reggae
    |        ├ rock
    

    Args:
        path        (str): path to data folder
        reg_exp     (str): regex pattern for file names
        n_class     (int): number of output class 
        t_windows   (int, optional: 6) : time frame of audio files
        sample_rate (int, optional: 22050) : audio sampling frequency
        resample    (int, optional: 15) : number of sampling on single audio
        split ratio (list, optional: [80, 10, 10]): train ,dev, and test ratios

    Methods:
        load(self):
            loads all audio files under data folder and returns them in a pandas dataframe
        save_split(self, save)
            calls self.load() first to get dataframe, splits dataframe into train, development,
            and test sets with respect to given ratio, and save them as npz file under data 
            folder if save argument is passed True
    """
    def __init__(self, path: str, reg_exp: str,  n_class: int, 
                 t_windows: float = 6.0, sample_rate: int = 22050,
                 resample: int = 15, split_ratio: list = [80, 10, 10]):
        self.path = path
        self.tw = t_windows
        self.rs = resample
        self.re = reg_exp
        self.sr = sample_rate
        self.nc = n_class
        self.ratio = split_ratio

    def load(self):
        """Loads all audio files under subdirectories of ~/data folder
        using librosa library. It resamples a single audio file to augment
        the number of total samples and returns pandas dataframe with an index
        of filenames and columns of sampling rate (frequencies)

        Parameters: 
            self (onject): AudioProcess object 
          
        Returns: 
            musics: A pandas dataframe with a 
                                        (number of audio file * resample) rows
                                        (sampling_rate * t_windows) columns
        """
        wav_files = self.file_list
        form_df = {'step_'+str(idx): np.empty(len(wav_files)*self.rs, dtype=np.float32) \
                    for idx in np.arange(1, self.tw*self.sr+1)}
        musics = pd.DataFrame(form_df)
        offset = np.tile(np.linspace(0, 30-self.tw-1, self.rs), (len(wav_files), 1))
        # offset = np.random.uniform(low=0.0, high=30-self.tw-np.finfo(float).eps, size=(len(wav_files), self.rs))
        with tqdm(total=len(wav_files), desc='Audio file are being processed') as pbar:
            for idx, file in enumerate(wav_files):
                # offset = np.random.rand() * (30-self.tw-np.finfo(float).eps)
                temp = [lbr.load(file, sr=self.sr, offset=offset[idx, jdx], duration=self.tw)[0] for jdx in range(self.rs)]
                # import pdb; pdb.set_trace()
                musics.iloc[self.rs*idx: self.rs*(idx+1), :] = np.array(temp)
                song = re.findall(self.re, file)[0]
                ind = dict(zip(range(self.rs*idx, self.rs*(idx+1)), 
                          ['_'.join([song.split('.')[0] + song.split('.')[1], str(x)]) for x in range(1, self.rs+1)]))
                musics.rename(index=ind, inplace=True)
                pbar.update(1)
        mapping = dict(zip(self.subdir, range(self.nc)))
        y = musics.index.str.extract(pat=r'([a-z]+)').replace(mapping).values
        musics['target'] = y
        return musics, offset

    def split(self, musics, save: bool = False):
        y = musics.loc[:, 'target'].values
        X = musics.iloc[:, :-1].values
        ts = (100 - self.ratio[0])/100
        X_train, X_dt, y_train, y_dt = train_test_split(X, y, test_size=ts, stratify=y)
        ts = self.ratio[-1] / sum(self.ratio[1:])
        X_dev, X_test, y_dev, y_test = train_test_split(X_dt, y_dt, test_size=ts, stratify=y_dt)
        if save:
            np.savez(os.path.join(self.path, 'train.npz'), X_train, y_train)
            np.savez(os.path.join(self.path, 'dev.npz'), X_dev, y_dev)
            np.savez(os.path.join(self.path, 'test.npz'), X_test, y_test)
        return X_train, y_train, X_dev, y_dev, X_test, y_test

    @property
    def subdir(self):
        """Returns folder names under ~/data folder"""
        return [f for f in os.listdir(self.path) if re.match(r'[a-z]+', f)]

    @property
    def file_list(self):
        """Returns list of file names with a .wav extension"""
        return glob(self.path + "/**/*.wav", recursive=True)