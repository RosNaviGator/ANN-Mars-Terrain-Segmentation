import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm


def unet_block(input_tensor, filters, kernel_size=3, activation='relu', stack=2, dropout_rate=0.3, name=''):
    """
    A block for downsampling in the U-Net. Includes Conv2D, BatchNormalization, Activation, 
    and optional Dropout layers, with Elastic Net regularization.
    """
    x = input_tensor
    for i in range(stack):
        x = tfkl.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-4),
            name=name + 'conv' + str(i + 1)
        )(x)
        x = tfkl.BatchNormalization(name=name + 'bn' + str(i + 1))(x)
        x = tfkl.Activation(activation, name=name + 'activation' + str(i + 1))(x)
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate, name=name + 'dropout' + str(i + 1))(x)
    return x


def se_block(input_tensor, reduction_ratio=16, name=''):
    """
    A Squeeze-and-Excitation block.
    """
    channels = input_tensor.shape[-1]
    se = tfkl.GlobalAveragePooling2D()(input_tensor)
    se = tfkl.Reshape((1, 1, channels))(se)
    se = tfkl.Dense(
        channels // reduction_ratio,
        activation='relu',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-4),
        name=name + 'dense1'
    )(se)
    se = tfkl.Dense(
        channels,
        activation='sigmoid',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-4),
        name=name + 'dense2'
    )(se)
    se = tfkl.Multiply(name=name + 'multiply')([input_tensor, se])
    return se


def dilated_conv_block(input_tensor, filters, dilation_rates, activation='relu', name=''):
    """
    A block with multiple dilated convolutions applied sequentially.
    """
    x = input_tensor
    for i, rate in enumerate(dilation_rates):
        x = tfkl.Conv2D(
            filters,
            kernel_size=3,
            dilation_rate=rate,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-4),
            name=name + f'dilated_conv{i+1}'
        )(x)
        x = tfkl.BatchNormalization(name=name + f'bn{i+1}')(x)
        x = tfkl.Activation(activation, name=name + f'activation{i+1}')(x)
    return x


def get_model(input_shape, num_classes, dropout_rate=0.3):
    """
    Constructs the Enhanced Base Unet model with U-Net architecture, SE blocks, 
    dilated convolutions, and Elastic Net regularization.
    """
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Downsampling path with SE blocks
    down_block_1 = unet_block(input_layer, 64, dropout_rate=dropout_rate, stack=3, name='down_block1_')
    down_block_1 = se_block(down_block_1, reduction_ratio=16, name='se_down_block1_')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = unet_block(d1, 128, dropout_rate=dropout_rate, stack=3, name='down_block2_')
    down_block_2 = se_block(down_block_2, reduction_ratio=16, name='se_down_block2_')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    down_block_3 = unet_block(d2, 256, dropout_rate=dropout_rate, stack=3, name='down_block3_')
    down_block_3 = dilated_conv_block(down_block_3, 256, dilation_rates=[1, 2, 4], name='dilated_down_block3_')
    d3 = tfkl.MaxPooling2D()(down_block_3)

    down_block_4 = unet_block(d3, 512, dropout_rate=dropout_rate, stack=3, name='down_block4_')
    down_block_4 = dilated_conv_block(down_block_4, 512, dilation_rates=[1, 2, 4], name='dilated_down_block4_')
    d4 = tfkl.MaxPooling2D()(down_block_4)

    # Bottleneck with SE and Dilated Convolutions
    bottleneck = unet_block(d4, 1024, dropout_rate=dropout_rate, stack=3, name='bottleneck_block_')
    bottleneck = dilated_conv_block(bottleneck, 1024, dilation_rates=[2, 4, 8], name='dilated_bottleneck_')
    bottleneck = se_block(bottleneck, reduction_ratio=16, name='se_bottleneck')

    # Upsampling path with SE blocks and dilated convolutions
    u1 = tfkl.UpSampling2D()(bottleneck)
    u1 = tfkl.Concatenate()([u1, down_block_4])
    u1 = unet_block(u1, 512, dropout_rate=dropout_rate, stack=3, name='up_block1_')
    u1 = se_block(u1, reduction_ratio=16, name='se_up_block1_')

    u2 = tfkl.UpSampling2D()(u1)
    u2 = tfkl.Concatenate()([u2, down_block_3])
    u2 = unet_block(u2, 256, dropout_rate=dropout_rate, stack=3, name='up_block2_')
    u2 = dilated_conv_block(u2, 256, dilation_rates=[1, 2, 4], name='dilated_up_block2_')

    u3 = tfkl.UpSampling2D()(u2)
    u3 = tfkl.Concatenate()([u3, down_block_2])
    u3 = unet_block(u3, 128, dropout_rate=dropout_rate, stack=3, name='up_block3_')
    u3 = se_block(u3, reduction_ratio=16, name='se_up_block3_')

    u4 = tfkl.UpSampling2D()(u3)
    u4 = tfkl.Concatenate()([u4, down_block_1])
    u4 = unet_block(u4, 64, dropout_rate=dropout_rate, stack=3, name='up_block4_')
    u4 = dilated_conv_block(u4, 64, dilation_rates=[1, 2], name='dilated_up_block4_')

    # Output Layer
    output_layer = tfkl.Conv2D(
        num_classes,
        kernel_size=1,
        padding='same',
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-4),
        name='output_layer'
    )(u4)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='Unet_elestiNet')
    return model
