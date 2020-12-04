# import glob
import os
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
# from skimage import img_as_float
# from skimage.io import imread, imsave
# from skimage.transform import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class VehicleDataset():
    def __init__(self, args):
        data_train = []

        for category in sorted(os.listdir(args.train)):
            for file in sorted(os.listdir(os.path.join(args.train, category))):
                data_train.append((category, os.path.join(args.train, category, file)))

        train_df = pd.DataFrame(data_train, columns=['class', 'file_path']).sample(frac=1.0)
        st.dataframe(train_df.head())
        profile = ProfileReport(train_df)
        if st.button("Generate Report"):
            st_profile_report(profile)

        # Data Generator for the train data
        data_generator = ImageDataGenerator(rescale=1. / 255,
                                            validation_split=0.2,
                                            horizontal_flip=True,
                                            rotation_range=10,
                                            width_shift_range=.1,
                                            height_shift_range=.1)
        # training data
        self.train_generator = data_generator.flow_from_dataframe(
            dataframe=train_df,
            directory=None,
            x_col='file_path',
            y_col='class',
            has_ext=False,
            subset="training",
            batch_size=args.batch_size,
            seed=42,
            shuffle=True,
            class_mode='categorical',
            target_size=(args.IMG_HEIGHT, args.IMG_WIDTH))

        # validation data
        self.validation_generator = data_generator.flow_from_dataframe(
            dataframe=train_df,
            directory=None,
            x_col='file_path',
            y_col='class',
            has_ext=False,
            subset="validation",
            batch_size=args.batch_size,
            seed=42,
            shuffle=True,
            class_mode='categorical',
            target_size=(args.IMG_HEIGHT, args.IMG_WIDTH))

# next information is not necessary for our project
#    def load_data(self, folder):
#        X = []  # Images go here
#        y = []  # Class labels go here
#        classes = []  # All class names go here

#        subdirectories = glob.glob(folder + "/*")

# Loop over all folders
#        for d in subdirectories:

# Find all files from this folder
#            files = glob.glob(d + os.sep + "*.jpg")

# Load all files
#            for name in files:

# Load image and parse class name
#                img = imread(name)
#                class_name = name.split(os.sep)[-2]

# Convert class names to integer indices:
#                if class_name not in classes:
#                    classes.append(class_name)

#                class_idx = classes.index(class_name)

#                X.append(img)
#                y.append(class_idx)

# Convert python lists to contiguous numpy arrays
#        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#        X = np.array(X)
#        y = np.array(y)
#        classes = np.array(classes)

#        return X, y, classes

#    def resize_test_data(self):
#        root = '../data/raw/vehicle/test/testset/'
#        new_path = '../data/raw/vehicle/test/scaled_test/'
#        os.mkdir(new_path)
#        i = 0
#        for file in sorted(os.listdir(root)):
#            img = imread(root + file)
#            res = resize(img, (200, 200))
#            imsave(new_path + os.sep + file, img_as_float(res))
#            i = i + 1
#            print(str(i) + ' images out of ' + str(len(os.listdir(root))) + ' processed')

#        print('Successfully resized')

#    def resize_train_data(self):
#        root = '../data/raw/vehicle/train/train/'
#        new_path = '../data/raw/vehicle/train/scaled_train/'
#        os.mkdir(new_path)
#        i = 0
# Rescale all files in each subdirectory
#        for category in sorted(os.listdir(root)):
#            os.mkdir(new_path + category)
#            for file in sorted(os.listdir(os.path.join(root, category))):
#                img = imread(root + category + os.sep + file)
#                res = resize(img, (200, 200))
#                imsave(new_path + os.sep + category + os.sep + file, img_as_float(res))
#                i = i + 1
#                print(str(i) + ' images processed')

#        print('Successfully resized')
