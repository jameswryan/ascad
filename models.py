import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Input,
    Conv1D,
    SeparableConv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    AveragePooling1D,
    BatchNormalization,
    Activation,
    Add,
    add,
)
from tensorflow.keras.optimizers import RMSprop, Adam

def alexnet(classes=256, input_dim=700):
    # From AlexNet design
    input_shape = (input_dim, 1)
    img_input = Input(shape=input_shape)
    # Block 0
    x = Conv1D(96, 11, activation="relu", padding="same", name="block0_conv0")(
        img_input
    )
    x = AveragePooling1D(2, strides=2, name="block0_pool")(x)
    # Block 1
    x = Conv1D(256, 9, activation="relu", padding="same", name="block1_conv0")(x)
    x = AveragePooling1D(2, strides=2, name="block1_pool")(x)
    # Block 2
    x = Conv1D(384, 9, activation="relu", padding="same", name="block2_conv0")(x)
    x = Conv1D(384, 9, activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv1D(256, 9, activation="relu", padding="same", name="block2_conv2")(x)
    x = AveragePooling1D(2, strides=2, name="block2_pool")(x)
    # Classification block
    x = Flatten(name="flatten")(x)
    x = Dense(2048, activation="relu", name="fc0")(x)
    x = Dense(2048, activation="relu", name="fc1")(x)
    x = Dense(classes, activation="softmax", name="predictions")(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name="alexnet-like")
    optimizer = RMSprop(lr=0.00001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


def vgg16(classes=256, input_dim=700):
    # From VGG16 design
    input_shape = (input_dim, 1)
    img_input = Input(shape=input_shape)
    # Block 0
    x = Conv1D(64, 11, activation="relu", padding="same", name="block0_conv0")(img_input)
    x = AveragePooling1D(2, strides=2, name="block0_pool")(x)
    # Block 1
    x = Conv1D(128, 11, activation="relu", padding="same", name="block1_conv0")(x)
    x = AveragePooling1D(2, strides=2, name="block1_pool")(x)
    # Block 2
    x = Conv1D(256, 11, activation="relu", padding="same", name="block2_conv0")(x)
    x = AveragePooling1D(2, strides=2, name="block2_pool")(x)
    # Block 3
    x = Conv1D(512, 11, activation="relu", padding="same", name="block3_conv0")(x)
    x = AveragePooling1D(2, strides=2, name="block3_pool")(x)
    # Block 4
    x = Conv1D(512, 11, activation="relu", padding="same", name="block4_conv0")(x)
    x = AveragePooling1D(2, strides=2, name="block4_pool")(x)
    # Classification block
    x = Flatten(name="flatten")(x)
    x = Dense(4096, activation="relu", name="fc0")(x)
    x = Dense(4096, activation="relu", name="fc1")(x)
    x = Dense(classes, activation="softmax", name="predictions")(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name="vgg16-like")
    optimizer = RMSprop(lr=0.00001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


def vgg19(classes=256, input_dim=700):
    # From VGG19 design
    input_shape = (input_dim, 1)
    img_input = Input(shape=input_shape)
    # Block 0
    x = Conv1D(64, 11, activation="relu", padding="same", name="block0_conv0")(
        img_input
    )
    x = Conv1D(64, 11, activation="relu", padding="same", name="block0_conv1")(x)
    x = AveragePooling1D(2, strides=2, name="block0_pool")(x)
    # Block 1
    x = Conv1D(128, 11, activation="relu", padding="same", name="block1_conv0")(x)
    x = Conv1D(128, 11, activation="relu", padding="same", name="block1_conv1")(x)
    x = AveragePooling1D(2, strides=2, name="block1_pool")(x)
    # Block 2
    x = Conv1D(256, 11, activation="relu", padding="same", name="block2_conv0")(x)
    x = Conv1D(256, 11, activation="relu", padding="same", name="block2_conv1")(x)
    x = AveragePooling1D(2, strides=2, name="block2_pool")(x)
    # Block 3
    x = Conv1D(512, 11, activation="relu", padding="same", name="block3_conv0")(x)
    x = Conv1D(512, 11, activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv1D(512, 11, activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv1D(512, 11, activation="relu", padding="same", name="block3_conv3")(x)
    x = AveragePooling1D(2, strides=2, name="block3_pool")(x)
    # Block 4
    x = Conv1D(512, 11, activation="relu", padding="same", name="block4_conv0")(x)
    x = Conv1D(512, 11, activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv1D(512, 11, activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv1D(512, 11, activation="relu", padding="same", name="block4_conv3")(x)
    x = AveragePooling1D(2, strides=2, name="block4_pool")(x)
    # Classification block
    x = Flatten(name="flatten")(x)
    x = Dense(4096, activation="relu", name="fc0")(x)
    x = Dense(4096, activation="relu", name="fc2")(x)
    x = Dense(classes, activation="softmax", name="predictions")(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name="vgg19-like")
    optimizer = RMSprop(lr=0.00001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model

def mobilenet(classes=256, input_dim=700):
    # From MobileNet design
    input_shape = (input_dim, 1)
    img_input = Input(shape=input_shape)
    # Block 0
    x = Conv1D(32, 9, strides=2, activation="relu", padding="same", name="input")(img_input)
    x = BatchNormalization()(x)

    # Keras SeparableConv1D combines Depthwise and Pointwise convolutions
    x = SeparableConv1D(32, 9, strides=2, depth_multiplier=2, activation="relu", padding="same", name="sep_conv0")(x)
    x = BatchNormalization()(x)

    x = SeparableConv1D(64, 9, strides=2, depth_multiplier=2, activation="relu", padding="same", name="sep_conv1")(x)
    x = BatchNormalization()(x)

    x = SeparableConv1D(128, 9, strides=1, depth_multiplier=1, activation="relu", padding="same", name="sep_conv2")(x)
    x = BatchNormalization()(x)

    x = SeparableConv1D(128, 9, strides=2, depth_multiplier=2, activation="relu", padding="same", name="sep_conv3")(x)
    x = BatchNormalization()(x)

    x = SeparableConv1D(256, 9, strides=1, depth_multiplier=1, activation="relu", padding="same", name="sep_conv4")(x)
    x = BatchNormalization()(x)

    x = SeparableConv1D(256, 9, strides=2, depth_multiplier=2, activation="relu", padding="same", name="sep_conv5")(x)
    x = BatchNormalization()(x)

    for i in range(6,11):
        x = SeparableConv1D(512, 9, strides=1, depth_multiplier=1, activation="relu", padding="same", name=f"sep_conv{i}")(x)
        x = BatchNormalization()(x)

    x = SeparableConv1D(512, 9, strides=2, depth_multiplier=2, activation="relu", padding="same", name="sep_conv11")(x)
    x = BatchNormalization()(x)

    x = SeparableConv1D(1024, 9, strides=2, depth_multiplier=1, activation="relu", padding="same", name="sep_conv12")(x)
    x = BatchNormalization()(x)

    x = AveragePooling1D(5)(x)

    # Classification block
    x = Flatten(name="flatten")(x)
    x = Dense(1024, activation="relu", name="fc0")(x)
    x = Dense(classes, activation="softmax", name="predictions")(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name="alexnet-like")
    optimizer = RMSprop(lr=0.00001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model

