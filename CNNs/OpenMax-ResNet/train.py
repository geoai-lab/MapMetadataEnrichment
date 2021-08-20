# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Add, Flatten, \
     Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import Progbar
from keras import backend as K

from load_image import minibatches, readCSV
from configuration import pretrained_weights, model_continent_path


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_model(img_rows, img_cols, color_type=3, num_classes=None):
    """
    Resnet 50 Model for Keras

    Model Schema is based on
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    ImageNet Pretrained Weights
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(8, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

    # Load ImageNet pre-trained data
    model.load_weights(pretrained_weights)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation=None, name='fc10')(x_newfc)
    x_newfc_softmax = Activation('softmax', name='fc10_activation')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc_softmax)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == '__main__':
    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224  # Resolution of inputs
    channel = 3
    num_classes = 8
    epochs = 50
    train_batch_len = 64
    valid_batch_len = 32
    Train_Images_Set = "ResNet/TrainImageNameList.csv"
    Valid_Images_Set = "ResNet/ValImageNameList.csv"

    # Training and validating process
    print("read image name list--start")
    X_train, y_train = readCSV(filepath=Train_Images_Set)
    X_val, y_val = readCSV(filepath=Valid_Images_Set)
    print("read image name list--end")

    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_classes)

    ## Train in epoches:
    for epoch in range(epochs):
        train_accuracy = 0
        valid_accuracy = 0
        train_rounds = 0
        valid_rounds = 0
        print("Epoch %d/%d turns" % (epoch, epochs))
        a = Progbar(int(len(X_train)/(train_batch_len))+1)
        for x_train_a, y_train_a in minibatches(X_train, y_train, batch_size=train_batch_len, shuffle=True, ratio=4):
            model.train_on_batch([x_train_a], y_train_a)
            a.update(train_rounds)
            train_rounds += 1
        a.update(train_rounds)

        print("----Begin to validate:----")
        for x_val_a, y_val_a in minibatches(X_val, y_val, batch_size=valid_batch_len, shuffle=True, ratio=1):
            results = model.evaluate(x_val_a, y_val_a)
            valid_accuracy += float(results[1])
            valid_rounds += 1

        valid_accuracy = valid_accuracy/valid_rounds

        print("accuracy on validation set is:" + str(valid_accuracy))
        if valid_accuracy > 0.99:
            break

        print(' ')

    # Save the model:
    model.save(model_continent_path)
