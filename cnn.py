import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def identify_gpu():
    print('FNC: identify_gpu')
    ''' If using a GPU it allows to enable memory growth '''
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print('Num GPUs available: {}'.format(len(physical_devices)))

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0],True)
    except Exception as e:
        print('GPU AVAILABILITY: {}'.format(e))
'''Going into data and randomly moving files to train,test,valid dirs '''
def organize_data():
    print('FNC: organize_data')
    os.chir('data/dogs-vs-cats')
    # Checks if the dir exists if not it creates those
    if os.path.isdir('train/dog') is False:
        os.makedirs('train/dog')
        os.makedirs('train/cat')
        os.makedirs('valid/dog')
        os.makedirs('valid/cat')
        os.makedirs('test/dog')
        os.makedirs('test/cat')

        ## random.sample returns list of random items from given set of data
        for c in random.sample(glob.glob('cat*'), 500):
            shutil.move(c,'train/cat')
        for c in random.sample(glob.glob('dog*'), 500):
            shutil.move(c,'train/dog')
        for c in random.sample(glob.glob('cat*'), 100):
            shutil.move(c,'valid/cat')
        for c in random.sample(glob.glob('dog*'), 100):
            shutil.move(c,'valid/dog')
        for c in random.sample(glob.glob('cat*'), 100):
            shutil.move(c,'test/cat')
        for c in random.sample(glob.glob('dog*'), 100):
            shutil.move(c,'test/dog')

def main():
    print('FNC: main')
    organize_data()

if __name__ == '__main__':
    main()
