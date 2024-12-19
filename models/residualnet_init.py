import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm


def residual_unet_block(input_tensor, filters, kernel_size=3, activation='relu', stack=2, dropout_rate=0, l2_reg=1e-4, name=''):
    """
    A block for the U-Net with residual connections. Includes Conv2D, BatchNormalization, Activation,
    optional Dropout layers, and L2 regularization.
    """
    kernel_regularizer = tf.keras.regularizers.L2(l2=l2_reg) if l2_reg and l2_reg > 0 else None
    x = input_tensor
    for i in range(stack):
        x = tfkl.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer='he_normal',
            name=name + 'conv' + str(i + 1)
        )(x)
        x = tfkl.BatchNormalization(name=name + 'bn' + str(i + 1))(x)
        x = tfkl.Activation(activation, name=name + 'activation' + str(i + 1))(x)
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate, name=name + 'dropout' + str(i + 1))(x)
    
    # Add residual connection
    shortcut = tfkl.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='he_normal',
        name=name + 'shortcut'
    )(input_tensor)
    x = tfkl.Add(name=name + 'residual_add')([x, shortcut])
    return x


def get_residual_unet(input_shape, num_classes, dropout_rate=0, l2_reg=1e-4):
    """
    Constructs a U-Net model with residual connections and transposed convolutions for upsampling.
    """
    kernel_regularizer = tf.keras.regularizers.L2(l2=l2_reg) if l2_reg and l2_reg > 0 else None
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Encoder (Downsampling Path)
    down_block_1 = residual_unet_block(input_layer, 32, dropout_rate=dropout_rate, stack=2, l2_reg=l2_reg, name='down_block1_')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = residual_unet_block(d1, 64, dropout_rate=dropout_rate, stack=2, l2_reg=l2_reg, name='down_block2_')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    down_block_3 = residual_unet_block(d2, 128, dropout_rate=dropout_rate, stack=2, l2_reg=l2_reg, name='down_block3_')
    d3 = tfkl.MaxPooling2D()(down_block_3)

    # Bottleneck with residual connection
    bottleneck = residual_unet_block(d3, 256, dropout_rate=dropout_rate, stack=2, l2_reg=l2_reg, name='bottleneck_block_')

    # Adjust d3 to match bottleneck's number of filters (256) using 1x1 convolution
    d3_residual = tfkl.Conv2D(
        256,  # Match the number of filters in the bottleneck
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal',
        name='bottleneck_residual_adjust'
    )(d3)

    # Add residual connection
    bottleneck_res = tfkl.Add(name='bottleneck_residual')([bottleneck, d3_residual])

    # Decoder (Upsampling Path)
    u1 = tfkl.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', name='up_transpose1')(bottleneck_res)
    u1 = tfkl.Concatenate(name='up_concat1')([u1, down_block_3])
    u1 = residual_unet_block(u1, 128, dropout_rate=dropout_rate, stack=2, l2_reg=l2_reg, name='up_block1_')

    u2 = tfkl.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', name='up_transpose2')(u1)
    u2 = tfkl.Concatenate(name='up_concat2')([u2, down_block_2])
    u2 = residual_unet_block(u2, 64, dropout_rate=dropout_rate, stack=2, l2_reg=l2_reg, name='up_block2_')

    u3 = tfkl.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', name='up_transpose3')(u2)
    u3 = tfkl.Concatenate(name='up_concat3')([u3, down_block_1])
    u3 = residual_unet_block(u3, 32, dropout_rate=dropout_rate, stack=2, l2_reg=l2_reg, name='up_block3_')

    # Final Output Layer
    output = tfkl.Conv2D(
        num_classes,
        kernel_size=1,
        padding='same',
        activation="softmax" if num_classes > 1 else "sigmoid",  # Use 'sigmoid' for binary classification
        kernel_regularizer=kernel_regularizer,
        name='unet_output_layer'
    )(u3)

    model = tf.keras.Model(inputs=input_layer, outputs=output, name='Residual_UNet_Model')
    return model
