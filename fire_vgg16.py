import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

def test_network(base_network, class_network, directory, sample_count):
    predicted_label_names = []
    actual_label_names = []
    file_names = []
    batch_size = 10

    datagen = ImageDataGenerator(rescale=1./255)    
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')    
    # Retrieve a list of all known class indexes.
    known_label_indices = (generator.class_indices)
    # Exchange the numeric indexs with textual lables in a dictionary.
    known_label_names = dict((v,k) for k,v in known_label_indices.items())        
    # Obtain the files names of each generated image and display the image
    # along with the name of the associated classification.
    file_names = generator.filenames[0:sample_count]
    # Add the base directory name to each file name to make a complete path.
    file_names = [os.path.join(directory, file_name) for file_name in file_names]

    # Process each batch of images provided by the generator through the specified network.
    i = 0
    for inputs_batch, labels_batch in generator:
        # Now transform the lables for all of the images into lable names.
        actual_label_names_batch = [known_label_names[k] for k in labels_batch.astype(int)]
        actual_label_names[i * batch_size : (i + 1) * batch_size] = actual_label_names_batch

        # Feed this batch of images to the base network to extract features using the pretrained
        # weights and biases.
        features_batch = base_network.predict(inputs_batch)
        # Flatten the data produced above by the base_network so that it may serve
        # as input to the densely connected class network.
        features_batch = np.reshape(features_batch, (batch_size, 4 * 4 * 512))

        # Save the generated features, labels, label names, and file names.        
        predicted_labels_batch = class_network.predict(features_batch)
        predicted_labels_int = np.rint(predicted_labels_batch)
        label_names = []
        for label_index in range(len(predicted_labels_int)):
            label_name_index = predicted_labels_int[label_index].item()
            label_names.append(known_label_names[label_name_index])
        
        predicted_label_names[i * batch_size : (i + 1) * batch_size] = label_names        

        # Do not allow the total number of processed images to exceed the total number of
        # samples.        
        i += 1
        if i * batch_size >= sample_count:
            break
        else:
            print(f"Test data count:{i * batch_size}, directory:{directory}")

    return predicted_label_names, actual_label_names, file_names        

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

        # Save the generated features, labels, label names, and file names.
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

def display_training_result(training_history):
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

def display_image_predictions(predictions, label_names, file_names):
#    w = 10
#    h = 10
    columns = 4
    rows = 4
    pages = 4

    # prep (x,y) for extra plotting
 #   xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
#    ys = np.abs(np.sin(xs))           # absolute of sine
    is_more_images = len(file_names) > 0

    for page_num in range(pages):
        if (not is_more_images):
            break
        fig = plt.figure(page_num + 1, figsize=(10, 12), dpi=150)        
        fig.subplots_adjust(hspace=.7)

        for i in range(columns * rows):
            image_index = page_num * columns * rows + i
            img = image.load_img(file_names[image_index], target_size=(150, 150))
    #        img = np.random.randint(10, size=(h,w))
            # create subplot and append to ax
            subplot = fig.add_subplot(rows, columns, i + 1)
            subplot.set_title("p" + predictions[image_index][0] + " l" + label_names[image_index][0] + " f" + file_names[image_index][-10:])  # set title
            subplot.imshow(img)       
    #        plt.imshow(img)

            # Stop displaying subplots if there are no additional images.
            if (image_index + 1 >= len(file_names)):
                is_more_images = False
                break

        results_file_name = "test_results_figure_{}.jpg".format(page_num + 1)
        print("Saving test results figure #{}, file name {}".format(page_num + 1, results_file_name))
        plt.savefig(results_file_name, format="jpg")

    # Display the images and their respective file names.
    # i = 0
    # for file_name in file_names:
    #     # Read the image and resize it
    #     img = image.load_img(file_name, target_size=(150, 150))

    #     # Display the array as an image.       
    #     plt.figure(i)
    #     plt.imshow(img)
    #     plt.title("P:" + predictions[i] + ", L:" + label_names[i] + ", F:" + file_names[i])
    #     i += 1

    # plt.show()    

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
    test_features, test_labels = extract_features(network, test_dir, 40)

    # Flatten the feature vectors to be used in the dense network which follows.
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (40, 4 * 4 * 512))

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
    display_training_result(history)

    # Display results generated from the test data.
    mse, mae = model.evaluate(test_features, test_labels)
    print("Test data accuracy: #", mae)

    # Apply the pretrained base network and the densely connected classifier to
    # the images in the test directory.
    predicted_label_names, actual_label_names, file_names = test_network(network, model, test_dir, 40)

    # Now display the predictions and the associated images in the test data.
    display_image_predictions(predicted_label_names, actual_label_names, file_names)

#    test_classes = model.predict_classes(test_features)
#    for i in range(len(test_classes)):
#        print("test #{}, class #{}, label #{}", i, test_classes[i], test_labels[i])

    # TODO-JYW: Use pinned tab to extract file names from the data generator.

if __name__== "__main__":
  main()
