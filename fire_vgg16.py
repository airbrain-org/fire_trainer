import sys
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

def main():
# TODO-JYW: Add named command line options: https://stackabuse.com/command-line-arguments-in-python/    
#    if len(sys.argv) < 2:
#        print_help()
    
    train_dir = 'd:/development/fire_data'

    conv_base = VGG16(weights='imagenet', include_top=False)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Freeze the VGG16 base to prevent updating the pretrained weight values.
    conv_base.trainable = False

    # TODO-JYW-LEFT-OFF: Examine each of the options for the ImageDataGenerator class.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)    


def print_help():
    pass
  
if __name__== "__main__":
  main()
