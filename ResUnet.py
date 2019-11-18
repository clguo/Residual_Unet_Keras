from keras.optimizers import *
from keras.models import *
from layer import  *
def ResUnet(input_size=(512, 512, 3), start_neurons=16):
    inputs = Input(input_size)
    conv1 = residual_block(inputs, start_neurons * 1, False)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = residual_block(pool1, start_neurons * 2, False)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)


    conv3 = residual_block(pool2, start_neurons * 4, False)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)


    # conv4 = residual_block(pool3, start_neurons * 8, False)
    # conv4 = residual_block(conv4, start_neurons * 8, True)
    # pool4 = MaxPooling2D((2, 2))(conv4)

    convm = residual_block(pool3, start_neurons * 8, False)
    convm = residual_block(convm, start_neurons * 8, True)

    # deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    # uconv4 = concatenate([deconv4, conv4])

    # uconv4 = residual_block(uconv4, start_neurons * 8, False)
    # uconv4 = residual_block(uconv4, start_neurons * 8, True)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = residual_block(uconv3, start_neurons * 4, False)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = residual_block(uconv2, start_neurons * 2, False)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = residual_block(uconv1, start_neurons * 1, False)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(input=inputs, output=output_layer)

    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

