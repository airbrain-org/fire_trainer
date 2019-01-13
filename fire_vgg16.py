import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

def extract_features(network, directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20

    # Create the generator that will process a number of images (batch_size)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    # Process each batch of images provided by the generator through the specified network.
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = network.predict(inputs_batch)
        # Save the generated features and labels into numpy array for return to the caller.
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        # Do not allow the total number of processed images to exceed the total number of
        # samples.        
        if i * batch_size >= sample_count:
            break
    return features, labels

def display_result(training_history):
    acc = training_history.history['acc']
    val_acc = training_history.history['val_acc']
    loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()  

def print_help():
    pass

def main():
# TODO-JYW: Add named command line options: https://stackabuse.com/command-line-arguments-in-python/    
#    if len(sys.argv) < 2:
#        print_help()

    base_dir = 'D:\\development\\screenshots'

    # Form the location of the traning data relative the base_dir specificed by the caller.
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # Load the pretrained network, except the top (last) layer used for classification.
    network = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(150, 150, 3))

    # Generate feature vectors for each of the images using the pretrained network.
    train_features, train_labels = extract_features(network, train_dir, 2000)
    validation_features, validation_labels = extract_features(network, validation_dir, 1000)
    test_features, test_labels = extract_features(network, test_dir, 1000)

    # Generate the dense network that will be used to classify the feature vectors 
    # generated above.
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim = 4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

    # Train the network using the feature vectors extracted from each training, validation, and
    # test image.
    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))
    display_result(history)

if __name__== "__main__":
  main()
