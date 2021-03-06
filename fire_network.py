# Copyright 2019 AirBrain.org
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this 
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimer in the documentation 
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import callbacks
import freeze

from keras import models
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.applications import VGG16
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K

# Training parameters:
detection_threshold = .90
num_training_images = 500
num_validation_images = 50
num_test_images = 40
model_fit_batch_size = 100
num_epochs = 80
x_image_pixels = 160
y_image_pixels = 160

# For VGG16:
# Note: Must also change the size of the features vector in extract_features()
# 150 x 150 = 4 * 4 * 512
# 160 x 160 = 5 * 5 * 512
# 512 x 512 = 16 * 16 * 512
# length_of_flattened_data = 5 * 5 * 512

# For MobileNet:
# Note: Must also change the size of the features vector in extract_features()
# 160 x 160 = 5 * 5 * 1024
length_of_flattened_data = 5 * 5 * 1024

# TODO-JYW: Add the CLI options referenced below
# TODO-JYW: Add code to check for the existence of directories mentioned below.

# Default training parameters, overriden with CLI options:
checkpoint_directory = "./checkpoint/"
tflite_directory = "./tflite/"
save_network_file = "fire_network.h5"
tensorboard_log_directory = "tensorboard"
early_stopping_improvement_epochs = 10
base_training_directory = '../screenshots'

def create_network(base_network, class_network):
    """
    Modify the base network by adding new dense output layers.  Then copy the weights
    of the class network, with the same dense output layers just added to the base network.

    
    @param base_network: pretrained image classifier
    @param class_network: network with densely connected layers, pretrained with output from the base_network

    @return network: single pretrained network, ready for testing    
    """

    # Connect the output layer specified to the input layer specified.
    x = base_network.output
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', input_dim = length_of_flattened_data)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid', name='final_output_layer')(x)

    # Compile the modified base network with the added dense layer.
    return_model = Model(inputs=base_network.input, outputs=x)
    return_model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

    # Copy the weights of the class network specified to the newly created class network layers.
    # Use the index corresponding to the last three layers.
    layer_index = -3
    for layer in class_network.layers:
        return_model_layer = return_model.layers[layer_index]
        return_model_layer.set_weights(layer.get_weights())
        layer_index += 1

    # Set all layers as untrainable since the weights have already been generated.    
    for layer in return_model.layers:
        layer.trainable = False

    return return_model

def test_network(network, directory, sample_count):
    batch_size = 10

    datagen = ImageDataGenerator(rescale=1./255)    
    generator = datagen.flow_from_directory(
        directory,
        target_size=(x_image_pixels, y_image_pixels),
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')    
    # Retrieve a list of all known class indexes.
    known_label_indices = (generator.class_indices)
    # Exchange the numeric indexes with textual lables in a dictionary.
    known_label_names = dict((v,k) for k,v in known_label_indices.items())        
    # Obtain the files names of each generated image and display the image
    # along with the name of the associated classification.
    file_names = generator.filenames[0:sample_count]
    # Add the base directory name to each file name to make a complete path.
    file_names = [os.path.join(directory, file_name) for file_name in file_names]

    # Process each batch of images provided by the generator through the specified network.
    predicted_label_names = []
    predicted_label_percent = []
    actual_label_names = []
    i = 0
    for inputs_batch, labels_batch in generator:
        # Now transform the lables for all of the images into lable names.
        actual_label_names_batch = [known_label_names[k] for k in labels_batch.astype(int)]
        actual_label_names[i * batch_size : (i + 1) * batch_size] = actual_label_names_batch

        # Save the generated features: label percents, label names, actual label names, and file names. 
        # Use the specified threshold to select the predicted class.
        predicted_labels_batch = network.predict(inputs_batch)
        predicted_label_percent[i * batch_size : (i + 1) * batch_size] = predicted_labels_batch        
        predicted_labels_int = [1 if label_percent >= detection_threshold else 0 for label_percent in predicted_labels_batch]

        label_names = []
        for label_index in range(len(predicted_labels_int)):
            label_name_index = predicted_labels_int[label_index]
            label_names.append(known_label_names[label_name_index])
        # Accumulate each batch of prediction names to return values to the caller.        
        predicted_label_names[i * batch_size : (i + 1) * batch_size] = label_names        

        # Do not allow the total number of processed images to exceed the total number of
        # samples.        
        i += 1
        if i * batch_size >= sample_count:
            break
        else:
            print("Test data count:" + str(i * batch_size) + ", directory:" + directory)

    return predicted_label_names, predicted_label_percent, actual_label_names, file_names        

def extract_features(network, directory, sample_count):
    # TODO-JYW: Represent the following shape in the training parameters section
    features = np.zeros(shape=(sample_count, 5, 5, 1024))
    labels = np.zeros(shape=(sample_count))
    batch_size = 10

    # Create the generator that will process a number of images (batch_size)
    datagen = ImageDataGenerator(rescale=1./255)    
    generator = datagen.flow_from_directory(
        directory,
        target_size=(x_image_pixels, y_image_pixels),
        batch_size=batch_size,
        class_mode='binary')

    # Process each batch of images provided by the generator through the specified network.
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = network.predict(inputs_batch)

        # Save the generated features, labels, label names, and file names.
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        # Do not allow the total number of processed images to exceed the total number of
        # samples.        
        i += 1
        if i * batch_size >= sample_count:
            break
        else:
            print("Feature count:" + str(i * batch_size) + ", directory:" + directory)

    return features, labels, generator

def display_training_result(training_history):
    pass
#    acc = training_history.history['acc']
#    val_acc = training_history.history['val_acc']
#    loss = training_history.history['loss']
#    val_loss = training_history.history['val_loss']

#    epochs = range(len(acc))

#    plt.plot(epochs, acc, 'bo', label='Training acc')
#    plt.plot(epochs, val_acc, 'b', label='Validation acc')
#    plt.title('Training and validation accuracy')
#    plt.legend()

#    plt.figure()

#    plt.plot(epochs, loss, 'bo', label='Training loss')
#    plt.plot(epochs, val_loss, 'b', label='Validation loss')
#    plt.title('Training and validation loss')
#    plt.legend()

#    plt.show()  

def print_help():
    pass

def display_image_predictions(predictions, membership, label_names, file_names):
    columns = 4
    rows = 4
    pages = 4

    is_more_images = len(file_names) > 0
    error_count = 0
    return_image_files = []

    for page_num in range(pages):
        if (not is_more_images):
            break
        fig = plt.figure(page_num + 1, figsize=(10, 12), dpi=150)        
        fig.subplots_adjust(hspace=.7)

        for i in range(columns * rows):
            image_index = page_num * columns * rows + i
            img = image.load_img(file_names[image_index], target_size=(150, 150))
            # create subplot and append to axis
            subplot = fig.add_subplot(rows, columns, i + 1)
            subplot.set_title("p" + predictions[image_index][0] + " l" + label_names[image_index][0] + " m" + str(membership[image_index][0])[0:4])  # set title
                # + " f" + file_names[image_index][-10:]            
            subplot.imshow(img)       
            # Keep track of the predictions which don't match the actual label names.
            if (predictions[image_index][0] != label_names[image_index][0]):
                error_count += 1
            # Stop displaying subplots if there are no additional images.
            if (image_index + 1 >= len(file_names)):
                is_more_images = False
                break

        results_file_name = "test_results_figure_{}.jpg".format(page_num + 1)
        print("Saving test results figure #" + str(page_num + 1) + ", file name " + results_file_name)
        plt.savefig(results_file_name, format="jpg")

        # Save the pyplot image filenames.
        return_image_files.append(results_file_name)

    print("Error percent: " + str(error_count / len(file_names)))    
    return return_image_files

def move_files(source, destintation, num_files):
    num_images = 0
    source_files = os.listdir(source)

    for file in source_files:
        os.rename(source + '/' + file, destintation + '/' + file)
        num_images += 1
        if (num_images >= num_files):
            break

def main():
    # Form the location of the training data relative the base_dir specificed by the caller.
    base_dir = base_training_directory
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # Make a validation directory, in case it is missing.
    if (not os.path.isdir(validation_dir)):
        os.mkdir(validation_dir)
        os.mkdir(validation_dir + '/nothing')
        os.mkdir(validation_dir + '/smoke')

    # Move some of the training data to the validation directory, if the validation directory
    # is empty.
    if (os.listdir(validation_dir + '/smoke') == []):
        move_files(train_dir + '/smoke', validation_dir + '/smoke', num_validation_images)
        move_files(train_dir + '/nothing', validation_dir + '/nothing', num_validation_images)
        print("Moved " + str(num_validation_images) + " images to " + validation_dir)

    # Load the pretrained network, except the top (last) layer used for classification.
    #network = VGG16(weights='imagenet',
    #        include_top=False,
    #        input_shape=(x_image_pixels, y_image_pixels, 3))
    network = MobileNet(weights='imagenet',
            include_top=False,
            input_shape=(x_image_pixels, y_image_pixels, 3))

    # Generate feature vectors for each of the images using the pretrained network.
    validation_features, validation_labels, data_generator = extract_features(network, validation_dir, num_validation_images)
    test_features, test_labels, data_generator = extract_features(network, test_dir, num_test_images)
    train_features, train_labels, data_generator = extract_features(network, train_dir, num_training_images)

    # Flatten the feature vectors to be used in the dense network which follows.
    train_features = np.reshape(train_features, (num_training_images, length_of_flattened_data))
    validation_features = np.reshape(validation_features, (num_validation_images, length_of_flattened_data))
    test_features = np.reshape(test_features, (num_test_images, length_of_flattened_data))

    # Generate the dense network that will be used to classify the feature vectors 
    # generated above.
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim = length_of_flattened_data))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid', name='final_output_layer'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

    # Train the network using the feature vectors extracted from each training, validation, and
    # test image.
    history = model.fit(train_features, train_labels,
                        epochs=num_epochs,
                        batch_size=model_fit_batch_size,
                        validation_data=(validation_features, validation_labels),
    # Setup of the callback executed at the end of every epoch, to check the performance on the test data.                        
                        callbacks=[callbacks.create_test_data_checkpoint(model, test_features, test_labels),
                                   callbacks.create_early_stopping(early_stopping_improvement_epochs),
                                   callbacks.create_model_checkpoint(checkpoint_directory + save_network_file),
                                   callbacks.create_tensorboard(tensorboard_log_directory)])
    display_training_result(history)

    # Indicate that this Keras network will only be used for inference.
    K.set_learning_phase(False)

    # Add the model trained above to the base network so that it can be saved as a single network.
    joined_network = create_network(network, model)
    joined_network.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

    # Display the names in the joined network:
    joined_network.summary()

    # Apply the pretrained base network and the densely connected classifier to
    # the images in the test directory.
    predicted_label_names, predicted_label_percent, actual_label_names, file_names = test_network(joined_network, test_dir, 40)

    # Save the last network processed by Keras to a file for later deployment.
    freeze.save_to_tflite(joined_network, tflite_directory, save_network_file, data_generator, True)

    # Now display the predictions and the associated images in the test data.
    display_image_predictions(predicted_label_names, predicted_label_percent, actual_label_names, file_names)

if __name__== "__main__":
  main()
