import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

# Import the TensorFlow backend to limit the number of threads used by Keras.
# from keras import backend as K

def extract_features(network, directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    batch_size = 10

    # Create the generator that will process a number of images (batch_size)
    datagen = ImageDataGenerator(rescale=1./255)    
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
        # Do not allow the total number of processed images to exceed the total number of
        # samples.        
        i += 1
        if i * batch_size >= sample_count:
            break
        else:
            print(f"Feature count:{i * batch_size}, directory:{directory}")
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

def display_image_predictions(network, directory):
    # Create the generator that will process a number of images (batch_size)
    datagen = ImageDataGenerator(rescale=1./255)    
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    generator.reset()

    # Run the network on the specified directory, and remember both the prediction
    # value and the associated prediction class.
    predictions = network.predict_generator(generator, verbose=1)
    predicted_class = np.argmax(predictions, axis=1)

    # Retrieve a list of all known class indexes.
    labels = (generator.class_indices)

    # Exchange the numeric indexs with textual lables in a dictionary.
    labels = dict((v,k) for k,v in labels.items())

    # Now transform the indexes of each prediction into the asscoicated 
    # class name and collect into the "predictions" array
    predictions = [labels[k] for k in predicted_class_indices]

    # Obtain the files names of each generated image and display the image
    # along with the name of the associated classification.
    filenames = generator.filenames


def main():
# TODO-JYW: Add named command line options: https://stackabuse.com/command-line-arguments-in-python/    
#    if len(sys.argv) < 2:
#        print_help()
    
    # Configure Keras to use a maximum of 32 threads.
    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_‌​parallelism_threads=‌32, inter_op_parallelism_threads=32)))

    # Form the location of the training data relative the base_dir specificed by the caller.
    base_dir = 'D:\\development\\screenshots'
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
    test_features, test_labels = extract_features(network, test_dir, 10)

    # Flatten the feature vectors to be used in the dense network which follows.
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (10, 4 * 4 * 512))

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
                        epochs=75,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))
    display_result(history)

    # Display results generated from the test data.
    mse, mae = model.evaluate(test_features, test_labels)
    print("Test data accuracy: #", mae)

//    test_classes = model.predict_classes(test_features)
//    for i in range(len(test_classes)):
//        print("test #{}, class #{}, label #{}", i, test_classes[i], test_labels[i])

    display_image_predictions()

    # TODO-JYW: Use pinned tab to extract file names from the data generator.

if __name__== "__main__":
  main()
