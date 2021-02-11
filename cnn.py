import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
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
import numpy as np
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

    # should give us 10 images with 10 corresponding labels bc batch size is 10
    imgs, labels = next(train_batches)
    # one hot encoded vectors [1,0] for cat, [0,1] dog which represent labels for each class
    return imgs, labels, train_batches, valid_batches, test_batches

    # This function will plot images in the form of a grid with 1 row and 10 columns where images are placed
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    # Color is distorted bc of the preprocessing function using the vgg16 model
    plt.show()

# build your model
def sequential_model(train_batches, valid_batches, test_batches):
    model = Sequential([
            Conv2D(filters=32, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (224,224,3)),
            MaxPool2D(pool_size=(2,2), strides=2),
            Conv2D(filters=32, kernel_size = (3,3), activation = 'relu', padding = 'same'),
            MaxPool2D(pool_size=(2,2), strides=2),
            Flatten(),
            Dense(units=2, activation = 'softmax'),
            ])
    print(model.summary())

    # prepare your model for training
    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics= ['accuracy'])
    # # train our model, didnt specify y because data has the corresponding labels within the generator itself
    # model.fit(x = train_batches, validation_data = valid_batches, epochs = 10, verbose = 2)

    '''output by 10th epoch
    Training data accuracy is 100 | validation is 55 %
     loss: 4.2481e-04 - accuracy: 1.0000 - val_loss: 4.0783 - val_accuracy: 0.5500

     --- we see overfitting, we should fine tune this since it did not generalize well
     '''
    return model

def train_model(model, train_batches, valid_batches):
    # train our model, didnt specify y because data has the corresponding labels within the generator itself
    model.fit(x = train_batches, validation_data = valid_batches, epochs = 10, verbose = 2)
    '''output by 10th epoch
    Training data accuracy is 100 | validation is 55 %
     loss: 4.2481e-04 - accuracy: 1.0000 - val_loss: 4.0783 - val_accuracy: 0.5500

     --- we see overfitting, we should fine tune this since it did not generalize well
     '''
    return model

def test_inference(model,test_batches):
    test_imgs, test_labels = next(test_batches)
    #plot_images(test_imgs)
    # direct unshuffled mapping for our matrix
    print(test_batches.classes)
    predictions = model.predict(x = test_batches, verbose = 0)
    # np arr of one hot encoded predictions [1,0] labels that it thinks it is class from index 0
    print(np.round(predictions))
    cm = confusion_matrix(y_true = test_batches.classes, y_pred= np.argmax(predictions, axis = -1))

    print(test_batches.class_indices)
    cm_plot_labels = ['cat','dog']
    plot_confusion_matrix(cm, cm_plot_labels,False,' Confusion Matrix',plt.cm.Blues)

def plot_confusion_matrix(cm, classes,normalize,title,cmap):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('normalized confusion_matrix')
    else:
        print('confusion_matrix without normalization')

    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
            horizontalalignment='center',
            color="white"if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')
    plt.show()


def main():
    print('FNC: main')
    img_arr, labels, train_batches, valid_batches, test_batches = process_data()
    #plot_images(img_arr)
    # build a model
    model = sequential_model(train_batches, valid_batches, test_batches)

    # Train model
    model = train_model(model, train_batches, valid_batches)

    # test img inference
    test_inference(model,test_batches)



if __name__ == '__main__':
    main()
    # obtain image ImageData
    # organize data on disk for reading
    # processed it for our CNN
    # build and train CNN
    # get train and validation accuracy
    # see how it performs with the test set
    # display a confusion matrix to analyze test data
