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



def recognize_with_timeoffset(sound_name_ex, sound_name_remote, samplerate, shorter_than_1min=True):
    
    '''
    （gclound上にある音声ファイルを）音声認識apiを使用して、結果の文字とタイムオフセットをcsvで保存する
    '''

    client = speech.SpeechClient()

    # todo 色々な拡張子が使えるようにする
    if sound_name_ex == "mp3":
        sys.exit()
        encoding = ""
    elif sound_name_ex == "wav":
        encoding = "LINEAR16"
    else:
        raise ValueError("音声ファイルの拡張子")


    all_words = ""
    word_time_list = []



    # １分より短い音声ファイルの場合
    if shorter_than_1min == True:

        results = client.recognize(
            audio = speech.types.RecognitionAudio(
                uri = sound_name_remote,
                ),
            config = speech.types.RecognitionConfig(
                encoding = encoding,
                language_code = "ja-JP",
                enable_word_time_offsets = True,
                sample_rate_hertz = samplerate,
                ),
        )
        
        a = results.ListFields()[0][1]

        for b in a:
            c = b.alternatives
            d = c[0]
            
            # 各単語の認識結果
            for word_info in d.words:
                
                print("="*20)

                # 各単語は'て|テ’の形で帰ってくることがある　
                word_with_katakana = word_info.word
                word = word_with_katakana.split("|")[0]
                
                # タイムスタンプを秒で表す
                start_time = word_info.start_time.seconds + float(word_info.start_time.nanos)/(10**9)
                end_time = word_info.end_time.seconds + float(word_info.end_time.nanos)/(10**9)
                
                all_words += word
                word_time_list.append((word, start_time, end_time))

                print("-------------")
                print("start time : {}".format(start_time))
                print(word)
                print("end time : {}".format(end_time))


    # １分より長い音声ファイルの場合
    elif shorter_than_1min == False:

        operation = client.long_running_recognize(
            audio = speech.types.RecognitionAudio(
                uri = sound_name_remote,
            ),
            config = speech.types.RecognitionConfig(
                encoding = encoding,
                language_code = "ja-JP",
                enable_word_time_offsets = True,
                sample_rate_hertz = samplerate,
            ),
        )
        
        op_result = operation.result()

        for result in op_result.results:
            
            for alternative in result.alternatives:
                
                print('=' * 20)
                print("confidence: {}".format(alternative.confidence))
                words = alternative.words
                
                for word_info in words:
                    
                    start_time = word_info.start_time.seconds + float(word_info.start_time.nanos)/(10**9)
                    end_time = word_info.end_time.seconds + float(word_info.end_time.nanos)/(10**9)
                    
                    word_with_katakana = word_info.word
                    word = word_with_katakana.split("|")[0]
                    
                    all_words += word
                    word_time_list.append((word, start_time, end_time))
                    
                    print("-------------")
                    print("start time : {}".format(start_time))
                    print(word)
                    print("end time : {}".format(end_time))

    
    return word_time_list
