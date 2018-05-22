# Import necessary modules
import os
import keras
import time
from pprint import pprint
from tqdm import tqdm
import numpy as np
import numpy.random as random
from scipy.misc import imresize, imread, imsave
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical

batch_size = 100
epochs = 50
input_shape = (120, 160, 1)

base_image_path = "ExtendedYaleB/"
people_paths = []
images = {}
files = []
num_classes = 0

for x in range(11, 40):
    
    if x == 14:
    
        pass
    
    else:
    
        people_paths.append(base_image_path + "yaleB" + str(x))

for i, x in enumerate(people_paths):
    
    num_classes += 1

    for file in tqdm(os.listdir(x)):
        
        if file.endswith(".pgm"):
            
            filename = x + "/" + file
            
            array = imread(filename)
            
            array = array.reshape(array.shape + (1,))
            
            files.append((array, i))

# Randomly shuffle training data
random.shuffle(files)

files = [list(x) for x in zip(*files)]


# the data, shuffled and split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    files[0], files[1], test_size=0.5)

#Convert python lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


# input data turned to floats and converted to numbers betwenn 0 and 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Function to build the nueral model
def build_model():
        
        global input_shape
        
        model = Sequential()
        
        model.add(Conv2D(20, kernel_size=(5, 5), strides=1,
                         activation='sigmoid', input_shape=input_shape))
        
        model.add(Dropout(0.2))
        
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        
        model.add(Dense(400, activation='sigmoid'))
        
        model.add(Dense(num_classes, activation='sigmoid'))
        
        return model

# Our input seeds
seeds = [10, 20, 30, 40, 50]
# seeds = [10, 20, 3]


for seed in seeds:
    
    old_time = time.time()
    
    random.seed(seed)


    model = build_model()

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    
    score = model.evaluate(x_test, y_test, verbose=0)

    with open("outputCNN.txt", "a+") as textfile:
        
        textfile.write("\n\n\nCNN")

        textfile.write("\nElapsed time: " + str((time.time() - old_time) / 60))
        
        textfile.write('\nTest loss:' + str(score[0]))
        
        textfile.write('\nTest accuracy:' + str(score[1]))
        
        textfile.write("\nSeed: " + str(seed))
