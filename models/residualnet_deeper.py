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

def squeeze_and_excitation(input_tensor, reduction_ratio=16, name=''):
    """
    Implements Squeeze-and-Excitation (SE) block to recalibrate channel-wise feature responses.
    """
    channels = input_tensor.shape[-1]
    se = tfkl.GlobalAveragePooling2D(name=name + 'global_avg_pool')(input_tensor)
    se = tfkl.Dense(channels // reduction_ratio, activation='relu', name=name + 'fc1')(se)
    se = tfkl.Dense(channels, activation='sigmoid', name=name + 'fc2')(se)
    se = tfkl.Multiply(name=name + 'scale')([input_tensor, tfkl.Reshape((1, 1, channels))(se)])
    return se


def atrous_spatial_pyramid_pooling(input_tensor, filters, l2_reg=1e-4, name=''):
    """
    Implements Atrous Spatial Pyramid Pooling (ASPP) for capturing multi-scale context.
    """
    kernel_regularizer = tf.keras.regularizers.L2(l2=l2_reg) if l2_reg > 0 else None

    conv1 = tfkl.Conv2D(filters, 1, padding='same', kernel_regularizer=kernel_regularizer, name=name + 'conv1')(input_tensor)
    conv1 = tfkl.BatchNormalization(name=name + 'bn1')(conv1)
    conv1 = tfkl.Activation('relu', name=name + 'activation1')(conv1)

    conv3 = tfkl.Conv2D(filters, 3, dilation_rate=3, padding='same', kernel_regularizer=kernel_regularizer, name=name + 'conv3')(input_tensor)
    conv3 = tfkl.BatchNormalization(name=name + 'bn3')(conv3)
    conv3 = tfkl.Activation('relu', name=name + 'activation3')(conv3)

    conv6 = tfkl.Conv2D(filters, 3, dilation_rate=6, padding='same', kernel_regularizer=kernel_regularizer, name=name + 'conv6')(input_tensor)
    conv6 = tfkl.BatchNormalization(name=name + 'bn6')(conv6)
    conv6 = tfkl.Activation('relu', name=name + 'activation6')(conv6)

    pooling = tfkl.GlobalAveragePooling2D(name=name + 'global_avg_pool')(input_tensor)
    pooling = tfkl.Reshape((1, 1, input_tensor.shape[-1]), name=name + 'reshape')(pooling)  # Reshape to (1, 1, channels)
    pooling = tfkl.Conv2D(filters, 1, padding='same', kernel_regularizer=kernel_regularizer, name=name + 'pool_conv')(pooling)
    pooling = tfkl.BatchNormalization(name=name + 'pool_bn')(pooling)
    pooling = tfkl.Activation('relu', name=name + 'pool_activation')(pooling)
    pooling = tfkl.UpSampling2D(size=(input_tensor.shape[1], input_tensor.shape[2]), interpolation='bilinear', name=name + 'upsample')(pooling)

    concatenated = tfkl.Concatenate(name=name + 'concat')([conv1, conv3, conv6, pooling])
    output = tfkl.Conv2D(filters, 1, padding='same', kernel_regularizer=kernel_regularizer, name=name + 'output_conv')(concatenated)
    return output

@tf.keras.utils.register_keras_serializable()
class AttentionGate(tfkl.Layer):
    def __init__(self, filters, l2_reg=1e-4, name='', **kwargs):
        super(AttentionGate, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.l2_reg = l2_reg
        self.name = name

    def build(self, input_shape):
        self.theta_x = tfkl.Conv2D(self.filters, kernel_size=1, strides=2, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg), name=self.name + 'theta_x')
        self.phi_g = tfkl.Conv2D(self.filters, kernel_size=1, padding='same',
                                  kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg), name=self.name + 'phi_g')
        self.psi = tfkl.Conv2D(1, kernel_size=1, padding='same',
                               kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg), name=self.name + 'psi')
        self.sigmoid = tfkl.Activation('sigmoid', name=self.name + 'sigmoid')
        self.upsample = tfkl.UpSampling2D(interpolation='bilinear', name=self.name + 'upsample')

    def call(self, inputs):
        input_tensor, gating_signal = inputs
        
        # Apply convolutions
        theta_x = self.theta_x(input_tensor)
        phi_g = self.phi_g(gating_signal)
        
        # Resize phi_g to match theta_x dimensions
        target_height = tf.shape(theta_x)[1]
        target_width = tf.shape(theta_x)[2]
        
        # Resize phi_g to match target height and width
        phi_g_resized = tf.image.resize(phi_g, size=(target_height, target_width))
        
        # Add the resized feature map to theta_x
        add_xg = tfkl.Add(name=self.name + 'add')([theta_x, phi_g_resized])
        act_xg = tfkl.Activation('relu', name=self.name + 'activation')(add_xg)
        
        # Generate attention map (psi)
        psi = self.psi(act_xg)
        psi = self.sigmoid(psi)
        
        # Resize psi map to match input_tensor dimensions
        psi_upsampled = tf.image.resize(psi, size=(tf.shape(input_tensor)[1], tf.shape(input_tensor)[2]))
        
        # Multiply the attention map with the input tensor
        attended_features = tfkl.Multiply(name=self.name + 'multiply')([input_tensor, psi_upsampled])
        return attended_features

    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({
            'filters': self.filters,
            'l2_reg': self.l2_reg,
            'name': self.name
        })
        return config

def get_residual_unet_deep(input_shape, num_classes, dropout_rate=0, l2_reg=1e-4):
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Encoder
    down_block_1 = residual_unet_block(input_layer, 64, dropout_rate=dropout_rate, stack=3, l2_reg=l2_reg, name='down_block1_')
    down_block_1 = squeeze_and_excitation(down_block_1, name='se_block1')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = residual_unet_block(d1, 128, dropout_rate=dropout_rate, stack=3, l2_reg=l2_reg, name='down_block2_')
    down_block_2 = squeeze_and_excitation(down_block_2, name='se_block2')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    down_block_3 = residual_unet_block(d2, 256, dropout_rate=dropout_rate, stack=3, l2_reg=l2_reg, name='down_block3_')
    down_block_3 = squeeze_and_excitation(down_block_3, name='se_block3')
    d3 = tfkl.MaxPooling2D()(down_block_3)

    # Bottleneck with ASPP
    bottleneck = atrous_spatial_pyramid_pooling(d3, 512, l2_reg=l2_reg, name='aspp_')

    # Decoder with Attention
    u1 = tfkl.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same', name='up_transpose1')(bottleneck)
    u1 = AttentionGate(256, l2_reg=l2_reg, name='att_gate1')([down_block_3, u1])
    u1 = tfkl.Concatenate(name='up_concat1')([u1, down_block_3])
    u1 = residual_unet_block(u1, 256, dropout_rate=dropout_rate, stack=3, l2_reg=l2_reg, name='up_block1_')

    u2 = tfkl.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', name='up_transpose2')(u1)
    u2 = AttentionGate(128, l2_reg=l2_reg, name='att_gate2')([down_block_2, u2])
    u2 = tfkl.Concatenate(name='up_concat2')([u2, down_block_2])
    u2 = residual_unet_block(u2, 128, dropout_rate=dropout_rate, stack=3, l2_reg=l2_reg, name='up_block2_')

    u3 = tfkl.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', name='up_transpose3')(u2)
    u3 = AttentionGate(64, l2_reg=l2_reg, name='att_gate3')([down_block_1, u3])
    u3 = tfkl.Concatenate(name='up_concat3')([u3, down_block_1])
    u3 = residual_unet_block(u3, 64, dropout_rate=dropout_rate, stack=3, l2_reg=l2_reg, name='up_block3_')

    # Output
    output = tfkl.Conv2D(
        num_classes,
        kernel_size=1,
        padding='same',
        activation="softmax" if num_classes > 1 else "sigmoid",
        kernel_regularizer=tf.keras.regularizers.L2(l2=l2_reg),
        name='output_layer'
    )(u3)

    model = tf.keras.Model(inputs=input_layer, outputs=output, name='Residual_UNet_Deep')
    return model
