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

from google.cloud import storage

'''
>>> bucket_name = "for_mp3"
>>> save_path = "music_files/music0.wav"

'''


def save_to_cloud_storage(original_file, bucket_name, save_path):
    
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # クラウドのバケットに保存
    blob2 = bucket.blob(save_path)
    blob2.upload_from_filename(filename=original_file)
    
    sound_path_remote = os.path.join("gs://", bucket_name, save_path)
    print("saved {}".format(sound_path_remote))