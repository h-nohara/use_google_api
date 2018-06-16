import os, sys
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


'''

'''


def recognize_and_to_csv(original_file, bucket_name, bucket_save_path, shorter_than_1min, df_save_path):
    
    '''
    音声認識apiを使って音声とタイムオフセットをデータフレームに保存する

    original_file : ローカルの音声ファイル
    bucket_name : クラウドのバケット名
    bucket_save_path : ローカルの音声ファイルの保存先（バケット以下のパス）
    shorter_than_1min : 音声ファイルが１分より短いか（短ければ同期音声認識、長ければ非同期音声認識）
    df_save_path : 認識結果のデータフレームを保存するパス

    >>> from recognize_with_timeoffset import recognize_and_to_csv
    >>> recognize_and_to_csv("./sound.wav", "for_mp3", "splits/sound3.wav", True, "./results/result3.csv")
    '''
    
    # path
    sound_name = original_file  # ローカルファイルのパス
    sound_name_base, sound_name_ex = os.path.basename(sound_name).split(".")
    sound_name_dir = os.path.dirname(sound_name)

    # データフレームの保存先のディレクトリ
    if not os.path.exists(os.path.dirname(df_save_path)):
        os.makedirs(os.path.dirname(df_save_path))


    # 音声ファイルを読み込み
    data, samplerate = sf.read(sound_name)
    print("data.shape --- {}".format(data.shape))
    print("data samplerate --- {}".format(samplerate))


    # もしチャンネル数が１じゃなかったらチャンネル数を１にして別名で保存
    if data.shape[1] != 1:

        onechannnel_sound_name = os.path.join(sound_name_dir, sound_name_base+"_onechannel"+"."+sound_name_ex)
        sound_name = onechannnel_sound_name

        sf.write(onechannnel_sound_name, data[:, 0], samplerate)  # チャンネル０を保存
        print("{} saved".format(onechannnel_sound_name))


    # クラウドのストレージにファイルを保存
    save_to_cloud_storage(original_file=sound_name, bucket_name=bucket_name, save_path=bucket_save_path)
    sound_name_remote = os.path.join("gs://", bucket_name, bucket_save_path)

    # 音声認識apiを使用する
    print("recognizing ...")
    word_time_list = recognize_with_timeoffset(
        sound_name_ex=sound_name_ex,
        sound_name_remote=sound_name_remote,
        samplerate=samplerate,
        shorter_than_1min=shorter_than_1min
        )
    print("reognize finished")

    # 認識された言葉と時間のcsvを保存する
    df = DataFrame({"start_time":[tup[1] for tup in word_time_list], "end_time":[tup[2] for tup in word_time_list]}, index=[tup[0] for tup in word_time_list])
    df = df.loc[:, ["start_time", "end_time"]]

    df.to_csv(df_save_path, encoding="utf-8")
    print("{} saved".format(df_save_path))

    print("all finished!")




def recognize_with_timeoffset(sound_name_ex, sound_name_remote, samplerate, shorter_than_1min=True):
    
    '''
    音声認識apiを使用して、結果の文字とタイムオフセットをcsvで保存する
    '''

    client = speech.SpeechClient()

    if sound_name_ex == "mp3":
        encoding = ""
    elif sound_name_ex == "wav":
        encoding = "LINEAR16"


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
            config=speech.types.RecognitionConfig(
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