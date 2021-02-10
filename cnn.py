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
    os.chdir('./data/dogs-vs-cats')
    print(os.getcwd())
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

    os.chdir('../../')

def process_data():
    train_path = './data/dogs-vs-cats/train'
    valid_path = './data/dogs-vs-cats/valid'
    test_path = './data/dogs-vs-cats/test'

    '''specify classes for cat,dog uses the name of the file as the label'''

    # processing data for keras api, flow_from_directory we pass data and specify how we want it to be processed
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes = ['cat','dog'], batch_size=10)

    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes = ['cat','dog'], batch_size=10)
    # shuffle is false for after training and validation to look at our results in a confusion_matrix to access unshuffled labels for our test set
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes = ['cat','dog'], batch_size=10,shuffle=False)
    try:
        # verify our processed images and amount per classes
        assert train_batches == 1000
        assert valid_batches == 200
        assert test_batches == 200
        assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes ==2
    except AssertionError as msg:
        print(msg)


    imgs, labels = next(train_batches)
    print(labels)
    return imgs, labels

    # This function will plot images in the form of a grid with 1 row and 10 columns where images are placed
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def main():
    print('FNC: main')
    img_arr, labels = process_data()
    plot_images(img_arr)

if __name__ == '__main__':
    main()
