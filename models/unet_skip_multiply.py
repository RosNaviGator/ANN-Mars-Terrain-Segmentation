import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm

def unet_block(input_tensor, filters, kernel_size=3, activation='relu', stack=2, dropout_rate=0.3, name=''):
    """
    A U-Net block composed of Conv2D, BatchNormalization, Activation, and Dropout layers.
    """
    # Initialise the input tensor
    x = input_tensor

    # Apply a sequence of Conv2D, Batch Normalization, Activation, and Dropout layers for the specified number of stacks
    for i in range(stack):
        x = tfkl.Conv2D(filters, kernel_size=kernel_size, padding='same', name=name + 'conv' + str(i + 1))(x)
        x = tfkl.BatchNormalization(name=name + 'bn' + str(i + 1))(x)
        x = tfkl.Activation(activation, name=name + 'activation' + str(i + 1))(x)
        if dropout_rate > 0:  # Apply dropout if a rate is specified
            x = tfkl.Dropout(dropout_rate, name=name + 'dropout' + str(i + 1))(x)

    # Return the transformed tensor
    return x


def get_model(input_shape, num_classes, seed=42, dropout_rate=0.3):
    
    # tf.random.set_seed(seed)  # Set seed for reproducibility
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Downsampling path
    down_block_1 = unet_block(input_layer, 32, dropout_rate=dropout_rate, name='down_block1_')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = unet_block(d1, 64, dropout_rate=dropout_rate, name='down_block2_')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    # Bottleneck
    bottleneck = unet_block(d2, 128, dropout_rate=dropout_rate, name='bottleneck')

    # Upsampling path using multiplication
    u1 = tfkl.UpSampling2D()(bottleneck)

    # Ensure down_block_2 and u1 have the same number of channels before multiplication
    u1 = tfkl.Conv2D(64, kernel_size=1, padding='same', name='align_channels_u1')(u1)  # Align the number of channels
    u1 = tfkl.Multiply(name='mul_skip1')([u1, down_block_2])  # Use multiplication instead of addition
    u1 = unet_block(u1, 64, dropout_rate=dropout_rate, name='up_block1_')

    u2 = tfkl.UpSampling2D()(u1)

    # Ensure down_block_1 and u2 have the same number of channels before multiplication
    u2 = tfkl.Conv2D(32, kernel_size=1, padding='same', name='align_channels_u2')(u2)  # Align the number of channels
    u2 = tfkl.Multiply(name='mul_skip2')([u2, down_block_1])  # Use multiplication instead of addition
    u2 = unet_block(u2, 32, dropout_rate=dropout_rate, name='up_block2_')

    # Output Layer
    output_layer = tfkl.Conv2D(num_classes, kernel_size=1, padding='same', activation="softmax", name='output_layer')(u2)

    # Create and return the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='Unet_skip_multiply')
    return model
