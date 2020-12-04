import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from classification_models.tfkeras import Classifiers
from keras.applications.inception_v3 import InceptionV3
import efficientnet.tfkeras as efn


def plot_accuracy(args, history):
    # Plot training & validation accuracy and lossvalues
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(args.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Training and Validation Loss')

    plt.show()


def model_simple(args):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(args.IMG_HEIGHT, args.IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(args.num_classes, activation='softmax')
    ])
    return model


def model_ResNet(args):
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights='imagenet',
                                                         pooling='max')

    model = Sequential()
    model.add(base_model)
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(args.num_classes, activation='softmax'))

    return model


def model_MobileNet(args):
    basemodel, _ = Classifiers.get('mobilenetv2')

    # build model
    base_model = basemodel(input_shape=(args.IMG_HEIGHT, args.IMG_WIDTH, 3),
                           weights='imagenet',
                           include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(args.num_classes, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    return model


def model_InceptionV3(args):
    base_model = InceptionV3(include_top=False,
                             weights="imagenet",
                             input_shape=(args.IMG_HEIGHT, args.IMG_WIDTH, 3),
                             pooling='max')
    model = Sequential()
    model.add(base_model)
    model.add(Dense(512, activation='relu'))
    model.add(Dense(args.num_classes, activation='softmax'))

    return model


def model_EfficientNet(args):
    base_model = efn.EfficientNetB7(include_top=False,
                                    input_shape=(args.IMG_HEIGHT, args.IMG_WIDTH, 3),
                                    weights='imagenet',
                                    pooling='max')
    model = Sequential()
    model.add(base_model)
    model.add(Dense(512, activation='relu'))
    model.add(Dense(args.num_classes, activation='softmax'))

    return model


def train(args, train_generator, validation_generator):

    # build a model
    print('building the best model..')
    model = model_MobileNet(args)
    print('done')

    # Compile the model
    print('compiling the model..')
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    print('done')

    # Class weights
    print('defining class weights..')
    class_weights_lst = class_weight.compute_class_weight('balanced',
                                                          np.unique(train_generator.classes),
                                                          train_generator.classes)
    class_weights = dict(zip(np.unique(train_generator.classes), class_weights_lst))
    print('done')

    # Stop training when a monitored quantity has stopped improving
    print('training will stop when a monitored quantity has stopped improving..')
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     patience=7,
                                                     verbose=1,
                                                     min_delta=1e-4)
    print('done')

    # If training does not improve after specific epochs, reduce the learning rate value by improving training
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.1,
                                                     patience=4,
                                                     verbose=1,
                                                     min_delta=1e-4)

    # Save the best model
    print('saving the best model..')
    file_path = f'../data/results/{model.name}.h5'
    best_model = tf.keras.callbacks.ModelCheckpoint(file_path,
                                                    save_best_only=True,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_weights_only=False)
    print('done')

    # Train model
    print('training model..')
    # history = model.fit(train_generator,
    #                    steps_per_epoch=21833 // args.batch_size,
    #                    epochs=args.epochs,
    #                    validation_data=validation_generator,
    #                    validation_steps=5458 // args.batch_size,
    #                    class_weight=class_weights,
    #                    callbacks=[earlyStopping, reduce_lr, best_model])
    print('done')