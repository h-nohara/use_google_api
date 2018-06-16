#!/user/bin/env python 
# -*- coding: utf-8 -*-

import os, sys
from os import makedirs
import glob2
import json
import subprocess
import collections
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

import cv2
import soundfile as sf

from google.cloud import speech

# use_google_apiの関数をインポート
USE_GOOGLE_API_PATH = "/Users/nohara/use_google_api/"
sys.path.append(USE_GOOGLE_API_PATH)
from storage_api.save_to_cloud_storage import save_to_cloud_storage
from speech_api.recognize_with_timeoffset import recognize_with_timeoffset

# music_movie_handlerの関数をインポート
MUSIC_MOVIE_HANDLERS_PATH = "/Users/nohara/music_movie_handlers/"
sys.path.append(MUSIC_MOVIE_HANDLERS_PATH)
from video_handlers.mp4_to_sound import mp4_to_sound


class Movie_Recognize:
    
    '''
    認識結果のデータフレームを作る関数
    '''
    
    def __init__(self, video_name):
        
        # 読み込んだ動画の情報
        self.video_name = video_name
        self.video_dir_name = os.path.dirname(video_name)
        self.video_ext_name = os.path.basename(video_name).split(".")[-1]
        if self.video_ext_name != "mp4":
            raise ValueError("the extention of video must mp4")

        # ベースネームと保存ディレクトリ
        self.base_name = os.path.basename(video_name).split(".")[0]
        self.save_dir = os.path.join(self.video_dir_name, self.base_name)
        if not os.path.exists(self.save_dir):
            makedirs(self.save_dir)

        # 音声ファイルの情報
        self.sound_name = None
        self.sound_data = None
        self.sound_samplerate = None

        # クラウドストレージの情報
        self.sound_name_remote = None
        self.bucket_name = None
        self.bucket_save_path = None

        # 認識結果
        self.word_time_list = None
        self.df = None
        self.df_name = None

    
    def save_wav(self, input_file=None, output_file=None, bitrate=128, channel=1):
        
        if input_file is None:
            input_file = self.video_name

        if output_file is None:
            output_file = os.path.join(self.save_dir, self.base_name+".wav")

        mp4_to_sound(input_file, output_file, bitrate, channel)

        self.sound_name = output_file


    def load_wav(self, sound_name=None):
        
        if sound_name is None:
            sound_name = self.sound_name

        # 音声ファイルを読み込み
        self.sound_data, self.sound_samplerate = sf.read(sound_name)
        print("sound_data.shape --- {}".format(self.sound_data.shape))
        print("sound_data samplerate --- {}".format(self.sound_samplerate))

        self.sound_name = sound_name


    def save_to_cloud_storage(self, original_file=None, bucket_name=None, bucket_save_path=None):
        
        # クラウドのストレージにファイルを保存
        if bucket_name is None:
            raise ValueError("define 'bucket name'!")
        if bucket_save_path is None:
            raise ValueError("define 'bucket_save_path name'!")
        
        save_to_cloud_storage(original_file=self.sound_name, bucket_name=bucket_name, save_path=bucket_save_path)

        self.sound_name_remote = os.path.join("gs://", bucket_name, bucket_save_path)
        self.bucket_name = bucket_name
        self.bucket_save_path = bucket_save_path

    
    def recognize(self, sound_name_ex, sound_name_remote, samplerate, shorter_than_1min):
        
        # 音声の長さを確認
        length_s = float(self.sound_data.shape[0]) / self.sound_samplerate
        if length_s < 60:
            shorter_than_1min = True
        else:
            shorter_than_1min = False

        sound_name_ex = os.path.basename(self.sound_name).split(".")[-1]


        self.word_time_list = recognize_with_timeoffset(
            sound_name_ex = sound_name_ex,
            sound_name_remote = self.sound_name_remote,
            samplerate = self.sound_samplerate,
            shorter_than_1min = shorter_than_1min,
            )

    def make_csv(self):
        
        # 認識された言葉と時間のcsvを保存する
        df = DataFrame({"start_time":[tup[1] for tup in self.word_time_list], "end_time":[tup[2] for tup in self.word_time_list]}, index=[tup[0] for tup in self.word_time_list])
        df = df.loc[:, ["start_time", "end_time"]]

        self.df_name = os.path.join(self.save_dir, "result.csv")
        df.to_csv(self.df_name, encoding="utf-8")
        print("{} saved".format(self.df_name))

        self.df = df




class Sound_Recognize:
    
    '''
    認識結果のデータフレームを作る関数
    '''
    
    def __init__(self, wav, result_csv=None):

        # 音声ファイルの情報
        self.sound_name = wav
        self.sound_data = None
        self.sound_samplerate = None

        ext_sound = os.path.basename(self.sound_name).split(".")[-1]
        if ext_sound != "wav":
            raise ValueError("the sound must be wav file.")  # 拡張子のチェック

        # クラウドストレージの情報
        self.sound_name_remote = None
        self.bucket_name = None
        self.bucket_save_path = None

        # 認識結果
        self.word_time_list = None
        self.df = None
        self.df_name = None

        # 保存するファイルのチェック
        if result_csv:
            if not os.path.exists(os.path.dirname(result_csv)):
                raise ValueError("the save folder doesnt exists.")  # 保存するディレクトリが存在するかチェック
            ext_csv = os.path.basename(result_csv).split(".")[-1]
            if ext_csv != "csv":
                raise ValueError("result file must be csv.")  # 拡張子をチェック
            self.result_csv = result_csv
            self.to_save = True
        elif result_csv is None:
            self.to_save = False

        self.result_DataFrame = None


        # 音声ファイルを読み込む
        self.load_wav()
            

    def load_wav(self):
        
        sound_name = self.sound_name

        # 音声ファイルを読み込み
        self.sound_data, self.sound_samplerate = sf.read(sound_name)
        print("sound_data.shape --- {}".format(self.sound_data.shape))
        print("sound_data samplerate --- {}".format(self.sound_samplerate))


    def save_to_cloud_storage(self, bucket_name=None, bucket_save_path=None):
        
        # クラウドのストレージにファイルを保存
        if bucket_name is None:
            raise ValueError("define 'bucket name'!")
        if bucket_save_path is None:
            raise ValueError("define 'bucket_save_path name'!")
        
        save_to_cloud_storage(original_file=self.sound_name, bucket_name=bucket_name, save_path=bucket_save_path)

        self.sound_name_remote = os.path.join("gs://", bucket_name, bucket_save_path)
        self.bucket_name = bucket_name
        self.bucket_save_path = bucket_save_path

    
    def recognize(self):
        
        # 音声の長さを確認
        length_s = float(self.sound_data.shape[0]) / self.sound_samplerate
        if length_s < 60:
            shorter_than_1min = True
        else:
            shorter_than_1min = False

        sound_name_ex = os.path.basename(self.sound_name).split(".")[-1]


        self.word_time_list = recognize_with_timeoffset(
            sound_name_ex = sound_name_ex,
            sound_name_remote = self.sound_name_remote,
            samplerate = self.sound_samplerate,
            shorter_than_1min = shorter_than_1min,
            )

    def make_csv(self):
        
        # 認識された言葉と時間のcsvを保存する
        df = DataFrame({"start_time":[tup[1] for tup in self.word_time_list], "end_time":[tup[2] for tup in self.word_time_list]}, index=[tup[0] for tup in self.word_time_list])
        df = df.loc[:, ["start_time", "end_time"]]

        if self.to_save:
            df.to_csv(self.result_csv, encoding="utf-8")
            print("{} saved".format(self.result_csv))

        self.result_DataFrame = df