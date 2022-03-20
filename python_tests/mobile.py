import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import sys

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (mobilenet_v2)

model = tf.keras.applications.MobileNetV2(
input_shape=(224,224,3),
alpha=1.0,
include_top=True,
weights="imagenet",
input_tensor=None,
pooling=None,
classes=1000,
classifier_activation="softmax",)

# Initialize model
mobilenetv2_model = mobilenet_v2.MobileNetV2(weights='imagenet')

# Load image in PIL format 244x244
filename = '/home/ricardo/EmbeddedNet/python_tests/test0.jpg'
original = load_img(filename, target_size=(224, 224))
print('PIL image size',original.size)
plt.imshow(original)
plt.show()

# Convert to numpy array 244x244x3
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)

# Add batchsize dimension 1x244x244x3
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))

# Preprocess
processed_image = mobilenet_v2.preprocess_input(image_batch.copy())

# Setup Data File
if (not os.path.exists("../data")):
    os.mkdir("../data")


def full_network():    
    # Predict
    predictions = mobilenetv2_model.predict(processed_image)
    label_mobile = decode_predictions(predictions)
    for prediction_id in range(len(label_mobile[0])):
        print(label_mobile[0][prediction_id])

    # Final Results for MobileNetV2 with ImageNet weights:
        #('n02123159', 'tiger_cat', 0.42557606)
        #('n02123045', 'tabby', 0.35453174)
        #('n02124075', 'Egyptian_cat', 0.070309244)
        #('n02127052', 'lynx', 0.007303906)
        #('n04074963', 'remote_control', 0.0032443653)


def layer(n):
    # Displays layer information
    model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",)

    print(model.layers[n].output)


def testlayer(n, l):
    import keras.backend as K
    model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",)
    
    func = K.function([model.layers[1].input], model.layers[n].output)
    conv_output = func([processed_image])
    
    print(conv_output.shape)

    output = np.single(conv_output[0,:,:,l])

    plt.imshow(output)
    plt.show()
    
    
def read(x):   
    file = open("../output.txt")
    fromC = np.loadtxt(file, delimiter=", ", usecols=range(x))
    plt.imshow(fromC)
    file.close()
    
    
def savepar(s):
    if s == 1:
        weight = "imagenet"
    elif s == 0:
        weight = None
    else:
        print("Invalid argument")
        sys.exit();
    
    model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    alpha=1.0,
    include_top=True,
    weights=weight,
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",)
    
    # Reset plots
    files_in_directory = os.listdir("../data")
    filtered_files = [file for file in files_in_directory if file.endswith(".dat")]
    for file in filtered_files:
        path_to_file = os.path.join("../data", file)
        os.remove(path_to_file)
    
    # Save Classification Labels
    print("Saving classification labels")
    import shutil
    shutil.copyfile("labels.txt", "../data/labels.txt")

    # Save Transfer Parameter
    print("Saving transfer parameter")
    with open('../data/transfer.csv', 'w') as outfile:
        outfile.write("1000 ")
    
    # Save Conv2D Weights
    indexC = [1, 7, 9, 16, 18, 24, 27, 34, 36, 42, 45, 51, 54, 61, 63, 69, 72, 78, 81, 87, 90, 96, 98, 104, 107, 113, 116, 123, 125, 131, 134, 140, 143, 149, 151]
    i = 1
    for x in indexC:
        print("Saving weights" + str(i) + ".csv")
        data = np.array(model.layers[x].get_weights())[0,:,:,:,:]
        name = '../data/weights' + str(i) + '.csv'
        i = i + 1
        with open(name, 'w') as outfile:
            for threeD_data_slice in data:
                for twoD_data_slice in threeD_data_slice:
                    np.savetxt(outfile, twoD_data_slice, fmt='%-1.8e')
    
    # Save Batch Normalization Weights
    indexBN = [2, 5, 8, 10, 14, 17, 19, 22, 25, 28, 32, 35, 37, 40, 43, 46, 49, 52, 55, 59, 62, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 99, 102, 105, 108, 111, 114, 117, 121, 124, 126, 129, 132, 135, 138, 141, 144, 147, 150, 152]
    i = 1
    for x in indexBN:
        print("Saving param" + str(i) + ".csv")
        data = np.array(model.layers[x].get_weights())
        name = '../data/param' + str(i) + '.csv'
        i = i + 1
        with open(name, 'w') as outfile:
            for twoD_data_slice in data: 
                    np.savetxt(outfile, twoD_data_slice, fmt='%-1.7e')
    
    # Save Depthwise Convolution Weights
    indexD = [4, 13, 21, 31, 39, 48, 58, 66, 75, 84, 93, 101, 110, 120, 128, 137, 146]
    i = 1
    for x in indexD:
        print("Saving dweights" + str(i) + ".csv")
        data = np.array(model.layers[x].get_weights())[0,:,:,:,0]
        name = '../data/dweights' + str(i) + '.csv'
        i = i + 1
        with open(name, 'w') as outfile:
            for threeD_data_slice in data:
                for twoD_data_slice in threeD_data_slice:
                    np.savetxt(outfile, twoD_data_slice, fmt='%-1.7e')
    
    # Save Fully Connected Weights/Biases
    print("Saving fc_w.csv")
    data = model.layers[155].get_weights()[0]
    with open('../data/fc_w.csv', 'w') as outfile:
        for twoD_data_slice in data:
            np.savetxt(outfile, twoD_data_slice, fmt='%-1.7e')
            
    print("Saving fc_b.csv")
    data = model.layers[155].get_weights()[1]
    with open('../data/fc_b.csv', 'w') as outfile:
        np.savetxt(outfile, data, fmt='%-1.7e')
    
    
def savebin(s):
    if s == 1:
        weight = "imagenet"
    elif s == 0:
        weight = None
    else:
        print("Invalid argument")
        sys.exit();
        
    model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    alpha=1.0,
    include_top=True,
    weights=weight,
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",)
    
    # Reset plots
    files_in_directory = os.listdir("../data")
    filtered_files = [file for file in files_in_directory if file.endswith(".dat")]
    for file in filtered_files:
        path_to_file = os.path.join("../data", file)
        os.remove(path_to_file)
    
    # Save Classification Labels
    print("Saving classification labels")
    import shutil
    shutil.copyfile("labels.txt", "../data/labels.txt")

    # Save Transfer Parameter
    print("Saving transfer parameter")
    with open('../data/transfer.csv', 'w') as outfile:
        outfile.write("1000 ")
    
    # Save All Parameters
    from array import array
    name = '../data/par.bin'
    if (os.path.exists(name)):
        os.remove(name)
    indexC = [1, 7, 9, 16, 18, 24, 27, 34, 36, 42, 45, 51, 54, 61, 63, 69, 72, 78, 81, 87, 90, 96, 98, 104, 107, 113, 116, 123, 125, 131, 134, 140, 143, 149, 151]
    indexBN = [2, 5, 8, 10, 14, 17, 19, 22, 25, 28, 32, 35, 37, 40, 43, 46, 49, 52, 55, 59, 62, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 99, 102, 105, 108, 111, 114, 117, 121, 124, 126, 129, 132, 135, 138, 141, 144, 147, 150, 152]
    indexD = [4, 13, 21, 31, 39, 48, 58, 66, 75, 84, 93, 101, 110, 120, 128, 137, 146]
    indexFC = [155]
    
    for i in range(156):
        
        if (i in indexC):
            print("Saving C  " + str(i))
            data = np.array(model.layers[i].get_weights())[0,:,:,:,:]
            with open(name, 'ab') as outfile:
                for threeD_data_slice in data:
                    for twoD_data_slice in threeD_data_slice:
                        for oneD_data_slice in twoD_data_slice:
                            float_array = array('f', oneD_data_slice)
                            float_array.tofile(outfile)
            
        elif (i in indexBN):
            print("Saving BN " + str(i))
            data = np.array(model.layers[i].get_weights())
            with open(name, 'ab') as outfile:
                for twoD_data_slice in data: 
                    #for oneD_data_slice in twoD_data_slice:
                        float_array = array('f', twoD_data_slice)
                        float_array.tofile(outfile)
                            
        elif (i in indexD):
            print("Saving D  " + str(i))
            data = np.array(model.layers[i].get_weights())[0,:,:,:,0]
            with open(name, 'ab') as outfile:
                for threeD_data_slice in data:
                    for twoD_data_slice in threeD_data_slice:
                        #for oneD_data_slice in twoD_data_slice:
                            float_array = array('f', twoD_data_slice)
                            float_array.tofile(outfile)
                            
        elif (i in indexFC):
            print("Saving FC weights")
            data = model.layers[i].get_weights()[0]
            with open(name, 'ab') as outfile:
                for twoD_data_slice in data:
                    #for oneD_data_slice in twoD_data_slice:
                        float_array = array('f', twoD_data_slice)
                        float_array.tofile(outfile)
                            
            print("Saving FC biases")
            data = model.layers[i].get_weights()[1]
            with open(name, 'ab') as outfile:
                float_array = array('f', data)
                float_array.tofile(outfile)    
             
                
def saveimages(s, rang=2600):
    if (rang % 2) != 0:  
        print("Range must be even")
        sys.exit();
    if s == 0:   # Save train images
        if (rang < 1 or rang > 2600):
            print("Train range must be between 1 and 2600")
            sys.exit();
        z = 0
        save = ''
    elif s == 1: # Save test images
        if (rang == 2600):
            rang = 400
            print("Assuming default = 400 images")
        if (rang < 1 or rang > 400):
            print("Test range must be between 1 and 400")
            sys.exit();
        z = 1300
        save = 'test'
    else:
        print("Invalid argument")
        sys.exit();
    
    import random
    random.seed(72)
    num = random.sample(range(rang), rang) # 2600 train / 400 test
    
    for x in range(rang):
        
        if (num[x] < rang/2):
            # Load image in PIL format 244x244
            filename = 'yes/y' + str(num[x] + z) + '.jpg'
            original = load_img(filename, target_size=(224, 224))

            # Convert to numpy array 244x244x3
            numpy_image = img_to_array(original)
            plt.imshow(np.uint8(numpy_image))
            plt.show()
        
            # Add batchsize dimension 1x244x244x3
            image_batch = np.expand_dims(numpy_image, axis=0)
            plt.imshow(np.uint8(image_batch[0]))
            
            # Preprocess
            processed_image = mobilenet_v2.preprocess_input(image_batch.copy())
            
            label = "1 "
        else:
            # Load image in PIL format 244x244
            filename = 'no/no' + str(num[x] - int(rang/2) + z) + '.jpg'
            original = load_img(filename, target_size=(224, 224))
            plt.imshow(original)
            plt.show()
            
            # Convert to numpy array 244x244x3
            numpy_image = img_to_array(original)
            plt.imshow(np.uint8(numpy_image))
            plt.show()
            
            # Add batchsize dimension 1x244x244x3
            image_batch = np.expand_dims(numpy_image, axis=0)
            plt.imshow(np.uint8(image_batch[0]))
            
            # Preprocess
            processed_image = mobilenet_v2.preprocess_input(image_batch.copy())
    
            label = "0 "
        
        # Save Images and Labels
        print("Saving image " + str(x) + " (" + str(num[x]+z) + ")")
        data = processed_image[0,:,:,:]
        with open('../data/image' + save + str(x) + '.csv', 'w') as outfile:
            for threeD_data_slice in data:
                for twoD_data_slice in threeD_data_slice:
                    np.savetxt(outfile, twoD_data_slice, fmt='%-1.7e')
                
        with open('../data/label' + save + str(x) + '.csv', 'w') as outfile:
            outfile.write(label)        


def savetestimage(x, l):
    
    # Load image in PIL format 244x244
    filename = 'test' + str(x) + '.jpg'
    original = load_img(filename, target_size=(224, 224))

    # Convert to numpy array 244x244x3
    numpy_image = img_to_array(original)
    plt.imshow(np.uint8(numpy_image))
    plt.show()
        
    # Add batchsize dimension 1x244x244x3
    image_batch = np.expand_dims(numpy_image, axis=0)
    plt.imshow(np.uint8(image_batch[0]))
            
    # Preprocess
    processed_image = mobilenet_v2.preprocess_input(image_batch.copy())

    # Save Images and Labels
    print("Saving test image")
    data = processed_image[0,:,:,:]
    with open('../data/imagetest400.csv', 'w') as outfile:
        for threeD_data_slice in data:
            for twoD_data_slice in threeD_data_slice:
                np.savetxt(outfile, twoD_data_slice, fmt='%-1.7e')
                
    with open('../data/labeltest400.csv', 'w') as outfile:
        outfile.write(str(l) + " ")
    
    
def train():
    import keras.backend as K

    newmodel = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    alpha=1.0,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",)

    func = K.function(newmodel.layers[1].input, newmodel.layers[155].output)
    conv_output = func([processed_image])
    y_pred = np.single(conv_output)
    
    #y_true = np.random.randint(low = 0, high = 999, size = batch_size)
    y_true = np.array([282])

    scce = tf.keras.losses.SparseCategoricalCrossentropy() 
    loss = scce(y_true, y_pred).numpy()
    print(loss)
    
    newmodel.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.RMSprop(
                                learning_rate=0.045, rho=0.9,
                                momentum=0.9, epsilon=1e-07,
                                centered=False, name='RMSprop'),
                      metrics=['accuracy'])
    newmodel.train_on_batch(x=processed_image, y=y_true, sample_weight=None, class_weight=None)

    print(newmodel.layers[155].get_weights())
