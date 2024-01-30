# Imports the Google Cloud Translation library
from google.cloud import translate_v2 as translate
import os
import pickle
import json
import tarfile
from tqdm import tqdm
import glob
import shutil
from chatpgt import predict_new
from multiprocessing.pool import Pool as Pool

