import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

matplotlib.use('TkAgg')

from keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import preprocessing


def test_dataframe(args):
    print('building dataset to predict..')
    # building test dataset
    data_test = []
    id_line = []

    count = 0
    for file in sorted(os.listdir(os.path.join(args.test))):
        data_test.append((os.path.join(args.test, file)))
        id_line.append(count)
        count += 1

    test_df = pd.DataFrame(data_test, columns=['file_path'])
    print('done')
    return test_df


def predicted_data(args, test_dataframe):
    # Load the best model
    print('loading model trained..')
    network = load_model(f'./data/results/complete_model_{args.model}.h5')
    network.summary()
    print('done')

    # Data Generator for predict data
    test_generator = ImageDataGenerator(rescale=1. / 255.)

    # predict data
    predict_generator = test_generator.flow_from_dataframe(
        dataframe=test_dataframe,
        x_col='file_path',
        y_col=None,
        batch_size=32,
        shuffle=False,
        class_mode=None,
        target_size=(224, 224))

    print("adapting the data to predict...")
    steps = len(test_dataframe) // 2
    print("predicting images through model trained..")
    proof = np.array(network.predict(predict_generator,
                                     steps=steps,
                                     verbose=1,
                                     workers=10,
                                     max_queue_size=64))
    return proof


def predicted_label(test_dataframe, predicted_data):
    print("extracting labels of the model trained..")
    label_names = ['Ambulance', 'Bicycle', 'Boat', 'Bus', 'Car', 'Helicopter', 'Limousine',
                   'Motorcycle', 'PickUp', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']
    print("done")

    print("assign predicted label to each image")
    class_names = []
    for j in range(len(test_dataframe)):
        b = np.argmax(predicted_data[j], axis=0)
        class_names.append(label_names[b])
    print("done")

    print("transform labels to array..")
    le = preprocessing.LabelEncoder()
    le.fit(class_names)
    label_data = le.transform(class_names)
    print("done")
    return label_data


def plot_image_pred(args,
                    labels: np.ndarray,
                    preds: np.ndarray,
                    indexes: list,
                    class_names: list = None,
                    figsize: tuple = (8, 4)):
    for index in indexes:
        predicted_probas = preds[index]
        true_index = labels[index]
        predicted_index = np.argmax(predicted_probas)

        if class_names:
            true_class = class_names[true_index]
            predicted_class = class_names[predicted_index]

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize)

        directory = sorted(os.listdir(args.test))
        axes[0].imshow(mpimg.imread(f"./data/raw/vehicle/test/proof/{directory[index]}"))
        axes[0].axis('off')
        axes[0].set_title(f"predicted class: {predicted_class if class_names else predicted_index}",
                          color='blue' if true_index == predicted_index else 'red')

        axes[1].barh(class_names if class_names else [str(i) for i in range(len(predicted_probas))],
                     predicted_probas, color='black')
        axes[1].get_children()[predicted_index].set_color('red')
        axes[1].get_children()[true_index].set_color('blue')
        axes[1].set_xlim(0, 1)
        axes[1].set_title("class probabilities")
        axes[1].set_xlabel("probability")
        axes[1].set_ylabel("class name")

        plt.tight_layout()
        plt.show()

# rest of data is before defined our scope. not valid for our project
# Load classes names
# data_dir = pathlib.Path('../data/raw/vehicle/train/train/')
# CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])


# def load_test_data(folder):
#    """
#    Load all test images from 'folder'.
#    """
#    X = []  # Images go here

# Find all test files from this folder
#    files = glob.glob(folder + os.sep + "*.jpg")
# Load all files
#    for name in files:
# Load image
#        img = plt.imread(name)
#        X.append(img)

# Convert python list to contiguous numpy array
#    X = np.array(X)

#    return X


# def generate_submit(args, test_generator):
#    filenames = test_generator.filenames
#    nb_samples = len(filenames)

#    best_model = tf.keras.models.load_model(args.model + '.h5')
#    predictions = best_model.predict_generator(test_generator, steps=nb_samples, verbose=1)
#    y_classes = predictions.argmax(axis=-1)
#    le = LabelEncoder().fit(CLASS_NAMES)
#    labels = list(le.inverse_transform(y_classes))

# Save the probabilities as a .csv file
# df = pd.DataFrame(data=predictions[1:, 1:],  # values
#              index=predictions[1:, 0],  # 1st column as index
#              columns=predictions[0, 1:])  # 1st row as the column names
# df.to_csv(args.model + '_probs.csv')
# print(df.head(10))
#    csv = 'Id, Ambulance, Barge, Bicycle, Boat, Bus, Car, Cart, Caterpillar, Helicopter, Limousine, Motorcycle,' \
#          'Segway, Snowmobile, Tank, Taxi, Truck, Van\n'
#    for n, row in enumerate(predictions):
#        csv += str(n)
#        for col in row:
#            csv += ',' + str(col)
#        csv += '\n'
#    open(args.model + '_probs.csv', 'w+').write(csv)

#    new_submission_path = args.model + ".csv"

#    with open(new_submission_path, "w") as fp:
#        fp.write("Id,Category\n")
#        for i, label in enumerate(labels):
#            fp.write("%d,%s\n" % (i, label))
#    print("Submission made!")
