import os
import re
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from collections import OrderedDict 
from librosa import stft, power_to_db
from librosa.feature import (mfcc, chroma_stft, chroma_cqt, chroma_cens, tonnetz, 
                            spectral_bandwidth, spectral_centroid, spectral_rolloff, 
                            zero_crossing_rate, spectral_contrast, rms, melspectrogram)


class Features():
    def __init__(self, path: str = os.path.join(os.getcwd(), 'data'), n_mfcc: int = 13, 
                n_chroma: int = 12, sample_rate: int = 22050, frame_length: int = 2048, hop_len: int = 512,
                t_windows: float = 6.0, resample: int = 15):
        self.path = path
        self.n_mf = n_mfcc
        self.n_ch = n_chroma
        self.sr = sample_rate
        self.n_fft = frame_length
        self.hl = hop_len
        self.tw = t_windows
        self.rs = resample
        self.seq_len = int(np.ceil(t_windows*sample_rate/hop_len))
        self.dim = n_mfcc + 3 * n_chroma + 6 + 4 + 7
        self.moments = ['max', 'mean', 'median', 'min', 'kurtosis', 'skew', 'std']
    
    def t_variant(self, *args, save: bool = False):
        if args:
            X_train, X_dev, X_test, y_train, y_dev, y_test = args
            shapex = X_train.shape[0], X_dev.shape[0], X_test.shape[0]
        else:
            npzfile = np.load(os.path.join(self.path, 'train.npz'), allow_pickle=True)
            X_train = npzfile['arr_0']
            y_train = npzfile['arr_1']
            npzfile = np.load(os.path.join(self.path, 'dev.npz'), allow_pickle=True)
            X_dev = npzfile['arr_0']
            y_dev = npzfile['arr_1']
            npzfile = np.load(os.path.join(self.path, 'test.npz'), allow_pickle=True)
            X_test = npzfile['arr_0']
            y_test = npzfile['arr_1']
            shapex = X_train.shape[0], X_dev.shape[0], X_test.shape[0]
        # Memory allocation
        X_a = np.concatenate((X_train, X_dev, X_test), axis=0)
        del X_train, X_dev, X_test
        feat = np.empty((X_a.shape[0], self.seq_len, self.dim))
        with tqdm(total=X_a.shape[0], desc='Feature Extraction Process') as pbar:
            for idx, row in enumerate(X_a):
                ct1, ct2 = 0, self.n_mf ### 0-13
                feat[idx, :, ct1:ct2] = np.transpose(mfcc(row, sr=self.sr, n_mfcc=self.n_mf, n_fft=self.n_fft, hop_length=self.hl))
                ct1 += self.n_mf; ct2 += self.n_ch ### 13-25
                feat[idx, :, ct1:ct2] = np.transpose(chroma_stft(row, sr=self.sr, hop_length=self.hl))
                ct1 += self.n_ch; ct2 += self.n_ch ### 25-37
                feat[idx, :, ct1:ct2] = np.transpose(chroma_cqt(y=row))
                ct1 += self.n_ch; ct2 += self.n_ch ### 37-49
                feat[idx, :, ct1:ct2] = np.transpose(chroma_cens(y=row))
                ct1 += self.n_ch; ct2 += 6 ### 49-55
                feat[idx, :, ct1:ct2] = np.transpose(tonnetz(y=row))
                ct1 += 6; ct2 += 1 ### 55-56
                feat[idx, :, ct1:ct2] = np.transpose(spectral_centroid(y=row, sr=self.sr))
                ct1 += 1; ct2 += 1 ### 56-57
                feat[idx, :, ct1:ct2] = np.transpose(spectral_bandwidth(y=row, sr=self.sr))
                ct1 += 1; ct2 += 1 ### 57-58
                feat[idx, :, ct1:ct2] = np.transpose(spectral_rolloff(y=row, sr=self.sr))
                ct1 += 1; ct2 += 7 ### 58-65
                feat[idx, :, ct1:ct2] = np.transpose(spectral_contrast(y=row))
                ct1 += 7; ct2 += 1 ### 65-66
                feat[idx, :, ct1:ct2] = np.transpose(zero_crossing_rate(y=row))
                pbar.update(1)
        feat_train = feat[:shapex[0], : ,:]
        feat_dev = feat[shapex[0]:sum(shapex[:2]), :, :]
        feat_test = feat[sum(shapex[:2]):, :, :]
        if save:
            np.savez(os.path.join(self.path, 'feat_train.npz'), feat_train, y_train)
            np.savez(os.path.join(self.path, 'feat_dev.npz'), feat_dev, y_dev)
            np.savez(os.path.join(self.path, 'feat_test.npz'), feat_test, y_test)
            return [(feat_train, y_train), (feat_dev, y_dev), (feat_test, y_test)]
        else:
            return [(feat_train, y_train), (feat_dev, y_dev), (feat_test, y_test)]

    def t_invariant(self, *args, save: bool = False):
        if args:
            X_train, X_dev, X_test, y_train, y_dev, y_test, file_list = args
            shapex = X_train.shape[0], X_dev.shape[0], X_test.shape[0]
        else:
            file_list = args
            npzfile = np.load(os.path.join(self.path, 'train.npz'), allow_pickle=True)
            X_train = npzfile['arr_0']
            y_train = npzfile['arr_1']
            npzfile = np.load(os.path.join(self.path, 'dev.npz'), allow_pickle=True)
            X_dev = npzfile['arr_0']
            y_dev = npzfile['arr_1']
            npzfile = np.load(os.path.join(self.path, 'test.npz'), allow_pickle=True)
            X_test = npzfile['arr_0']
            y_test = npzfile['arr_1']
            shapex = X_train.shape[0], X_dev.shape[0], X_test.shape[0]
        
        def create_df():
            ind = range(len(file_list)*self.rs)
            
            col_ind = OrderedDict(chroma_cens=12, chroma_cqt=12, chroma_stft=12, 
            mfcc=13, rmse=1, spectral_bandwidth=1, spectral_centroid=1, 
            spectral_contrast=7, spectral_rolloff=1, tonnetz=6, zcr=1)
            mult_ind = []
            for key, value in col_ind.items():
                for moment in self.moments:
                    for num in range(1, value+1):
                        mult_ind.append([key, moment, num])
            mult_ind = pd.MultiIndex.from_tuples(mult_ind, names=['feature', 'moment', 'number'])

            return pd.DataFrame(index=ind, columns=mult_ind)

        def get_features(X_a):
            feature_df = create_df()
            
            with tqdm(total=X_a.shape[0], desc='Feature Extraction Process') as pbar:
                for idx in range(X_a.shape[0]):
                    f = zero_crossing_rate(y=X_a[idx, :], frame_length=self.n_fft, hop_length=self.hl)
                    write_features(feature_df, f, idx, ['zcr'])
                    f = chroma_cqt(y=X_a[idx, :], n_chroma=12, sr=self.sr, hop_length=self.hl)
                    write_features(feature_df, f, idx, ['chroma_cqt'])
                    f = chroma_cens(y=X_a[idx, :], n_chroma=12, sr=self.sr, hop_length=self.hl)
                    write_features(feature_df, f, idx, ['chroma_cens'])
                    f = tonnetz(y=X_a[idx, :], sr=self.sr)
                    write_features(feature_df, f, idx, ['tonnetz'])
                    stft_vals = np.abs(stft(y=X_a[idx, :], n_fft=self.n_fft, hop_length=self.hl))
                    f = chroma_stft(S=stft_vals**2, n_chroma=self.n_ch)
                    write_features(feature_df, f, idx, ['chroma_stft'])
                    f = rms(S=stft_vals)
                    write_features(feature_df, f, idx, ['rmse'])
                    f = spectral_centroid(S=stft_vals)
                    write_features(feature_df, f, idx, ['spectral_centroid'])
                    f = spectral_bandwidth(S=stft_vals)
                    write_features(feature_df, f, idx, ['spectral_bandwidth'])
                    f = spectral_contrast(S=stft_vals, n_bands=6)
                    write_features(feature_df, f, idx, ['spectral_contrast'])
                    f = spectral_rolloff(S=stft_vals)
                    write_features(feature_df, f, idx, ['spectral_rolloff'])
                    mel = melspectrogram(sr=self.sr, S=stft_vals**2)
                    f = mfcc(S=power_to_db(mel), n_mfcc=self.n_mf)
                    write_features(feature_df, f, idx, ['mfcc'])
                    pbar.update(1)
            return feature_df

        def write_features(df, values, ind, col):
            mult_level_col = list(itertools.product(col, self.moments))
            df.loc[ind, mult_level_col[0]] = np.mean(values, axis=1)
            df.loc[ind, mult_level_col[1]] = np.std(values, axis=1)
            df.loc[ind, mult_level_col[2]] = stats.skew(values, axis=1)
            df.loc[ind, mult_level_col[3]] = stats.kurtosis(values, axis=1)
            df.loc[ind, mult_level_col[4]] = np.median(values, axis=1)
            df.loc[ind, mult_level_col[5]] = np.min(values, axis=1)
            df.loc[ind, mult_level_col[6]] = np.max(values, axis=1)
        
        # Memory allocation
        X_a = np.concatenate((X_train, X_dev, X_test), axis=0)
        feat = get_features(X_a)
        del X_a
        X_train = feat.iloc[:shapex[0], :]
        X_dev = feat.iloc[shapex[0]:sum(shapex[:2]), :]
        X_test = feat.iloc[sum(shapex[:2]):, :]
        X_train['target'] = y_train
        X_dev['target'] = y_dev
        X_test['target'] = y_test
        return (X_train, X_dev, X_test)