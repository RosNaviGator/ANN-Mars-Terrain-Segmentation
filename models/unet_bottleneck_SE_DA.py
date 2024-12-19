import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm

def unet_block(input_tensor, filters, kernel_size=3, activation='relu', stack=2, dropout_rate=0.3, name=''):
    x = input_tensor
    for i in range(stack):
        x = tfkl.Conv2D(filters, kernel_size=kernel_size, padding='same', name=name + 'conv' + str(i + 1))(x)
        x = tfkl.BatchNormalization(name=name + 'bn' + str(i + 1))(x)
        x = tfkl.Activation(activation, name=name + 'activation' + str(i + 1))(x)
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate, name=name + 'dropout' + str(i + 1))(x)
    return x

def se_block(input_tensor, reduction_ratio=16, name=''):
    channels = input_tensor.shape[-1]
    se = tfkl.GlobalAveragePooling2D()(input_tensor)
    se = tfkl.Reshape((1, 1, channels))(se)
    se = tfkl.Dense(channels // reduction_ratio, activation='relu', use_bias=False, name=name + 'dense1')(se)
    se = tfkl.Dense(channels, activation='sigmoid', use_bias=False, name=name + 'dense2')(se)
    se = tfkl.Multiply(name=name + 'multiply')([input_tensor, se])
    return se

def get_model(input_shape, num_classes, seed=42, dropout_rate=0.3):
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Downsampling path
    down_block_1 = unet_block(input_layer, 64, dropout_rate=dropout_rate, stack=3, name='down_block1_')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = unet_block(d1, 128, dropout_rate=dropout_rate, stack=3, name='down_block2_')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    down_block_3 = unet_block(d2, 256, dropout_rate=dropout_rate, stack=3, name='down_block3_')
    d3 = tfkl.MaxPooling2D()(down_block_3)

    down_block_4 = unet_block(d3, 512, dropout_rate=dropout_rate, stack=3, name='down_block4_')
    d4 = tfkl.MaxPooling2D()(down_block_4)

    # Bottleneck with both Dilated Convolutions and SE Block
    bottleneck = tfkl.Conv2D(1024, kernel_size=3, dilation_rate=2, padding='same', name='bottleneck_dilated_conv1')(d4)
    bottleneck = tfkl.BatchNormalization(name='bottleneck_bn1')(bottleneck)
    bottleneck = tfkl.Activation('relu', name='bottleneck_activation1')(bottleneck)

    bottleneck = tfkl.Conv2D(1024, kernel_size=3, dilation_rate=4, padding='same', name='bottleneck_dilated_conv2')(bottleneck)
    bottleneck = tfkl.BatchNormalization(name='bottleneck_bn2')(bottleneck)
    bottleneck = tfkl.Activation('relu', name='bottleneck_activation2')(bottleneck)

    bottleneck = se_block(bottleneck, reduction_ratio=16, name='se_bottleneck')

    # Upsampling path
    u1 = tfkl.UpSampling2D()(bottleneck)
    u1 = tfkl.Concatenate()([u1, down_block_4])
    u1 = unet_block(u1, 512, dropout_rate=dropout_rate, stack=3, name='up_block1_')

    u2 = tfkl.UpSampling2D()(u1)
    u2 = tfkl.Concatenate()([u2, down_block_3])
    u2 = unet_block(u2, 256, dropout_rate=dropout_rate, stack=3, name='up_block2_')

    u3 = tfkl.UpSampling2D()(u2)
    u3 = tfkl.Concatenate()([u3, down_block_2])
    u3 = unet_block(u3, 128, dropout_rate=dropout_rate, stack=3, name='up_block3_')

    u4 = tfkl.UpSampling2D()(u3)
    u4 = tfkl.Concatenate()([u4, down_block_1])
    u4 = unet_block(u4, 64, dropout_rate=dropout_rate, stack=3, name='up_block4_')

    # Output Layer
    output_layer = tfkl.Conv2D(num_classes, kernel_size=1, padding='same', activation="softmax", name='output_layer')(u4)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='Unet_bottleneck_SE_DA')
    return model
