import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm


def unet_block(input_tensor, filters, kernel_size=3, activation='relu', stack=2, dropout_rate=0.3, name=''):
    """
    A block for downsampling in the U-Net. Includes Conv2D, BatchNormalization, Activation, 
    and optional Dropout layers, with L2 regularization.
    """
    x = input_tensor
    for i in range(stack):
        x = tfkl.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
            name=name + 'conv' + str(i + 1)
        )(x)
        x = tfkl.BatchNormalization(name=name + 'bn' + str(i + 1))(x)
        x = tfkl.Activation(activation, name=name + 'activation' + str(i + 1))(x)
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate, name=name + 'dropout' + str(i + 1))(x)
    return x

def multi_scale_block(input_tensor, filters, kernel_size=3, dropout_rate=0.3, name=''):
    """
    Multi-scale block to extract features at multiple scales and include batch normalization and activation.
    
    Args:
        input_tensor: Input tensor.
        filters: Number of filters for the convolutional layers.
        kernel_size: Kernel size for the convolutions.
        dropout_rate: Dropout rate for regularization.
        name: Naming prefix for the layers.
    
    Returns:
        The output of the multi-scale block.
    """
    # 1x1 Convolution
    x1 = tfkl.Conv2D(filters, kernel_size=1, padding='same', name=f'{name}_conv1x1')(input_tensor)
    x1 = tfkl.BatchNormalization(name=f'{name}_bn1x1')(x1)
    x1 = tfkl.ReLU()(x1)
    if dropout_rate > 0:
        x1 = tfkl.Dropout(dropout_rate)(x1)

    # 3x3 Convolution
    x2 = tfkl.Conv2D(filters, kernel_size=3, padding='same', name=f'{name}_conv3x3')(input_tensor)
    x2 = tfkl.BatchNormalization(name=f'{name}_bn3x3')(x2)
    x2 = tfkl.ReLU()(x2)
    if dropout_rate > 0:
        x2 = tfkl.Dropout(dropout_rate)(x2)

    # 5x5 Convolution
    x3 = tfkl.Conv2D(filters, kernel_size=5, padding='same', name=f'{name}_conv5x5')(input_tensor)
    x3 = tfkl.BatchNormalization(name=f'{name}_bn5x5')(x3)
    x3 = tfkl.ReLU()(x3)
    if dropout_rate > 0:
        x3 = tfkl.Dropout(dropout_rate)(x3)

    # Concatenate all the outputs from different scales
    x = tfkl.Concatenate(name=f'{name}_concat')([x1, x2, x3])

    return x


def se_block(input_tensor, reduction_ratio=16, name=''):
    """
    A Squeeze-and-Excitation block with L2 regularization.
    """
    channels = input_tensor.shape[-1]
    se = tfkl.GlobalAveragePooling2D()(input_tensor)
    se = tfkl.Reshape((1, 1, channels))(se)
    se = tfkl.Dense(
        channels // reduction_ratio,
        activation='relu',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name=name + 'dense1'
    )(se)
    se = tfkl.Dense(
        channels,
        activation='sigmoid',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name=name + 'dense2'
    )(se)
    se = tfkl.Multiply(name=name + 'multiply')([input_tensor, se])
    return se
    
    
def cross_scale_attention_fusion(local_output, global_output, name='csa_fusion'):
    """
    Implements Cross-Scale Attention for feature fusion
    
    Args:
        local_output: Local branch feature map
        global_output: Global branch feature map
        name: Naming prefix for layers
    
    Returns:
        Fused feature map with cross-scale attention
    """
    # Concatenate features instead of simple addition
    concat_features = tfkl.Concatenate(name=f'{name}_concat')([local_output, global_output])
    
    # Channel attention mechanism
    channels = concat_features.shape[-1]
    
    # Global Average Pooling
    gap = tfkl.GlobalAveragePooling2D(name=f'{name}_gap')(concat_features)
    
    # Attention weights generation
    attention_weights = tfkl.Dense(
        channels, 
        activation='sigmoid', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name=f'{name}_attention_weights'
    )(gap)
    
    # Reshape attention weights
    attention_weights = tfkl.Reshape((1, 1, channels))(attention_weights)
    
    # Apply attention
    attended_features = tfkl.Multiply(name=f'{name}_attended_features')([concat_features, attention_weights])
    
    # Reduce channel dimension back to original size
    fused_output = tfkl.Conv2D(
        local_output.shape[-1],
        kernel_size=1,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name=f'{name}_fusion_conv'
    )(attended_features)
    
    return fused_output

@tf.keras.utils.register_keras_serializable()
class WarmUpDropPath(tfkl.Layer):
    """ WarmUpDropPath regularization (stochastic depth with warm-up). """

    def __init__(self, drop_prob_start=0.0, drop_prob_end=0.3, warmup_epochs=10, **kwargs):
        super(WarmUpDropPath, self).__init__(**kwargs)
        self.drop_prob_start = drop_prob_start
        self.drop_prob_end = drop_prob_end
        self.warmup_epochs = warmup_epochs

    def build(self, input_shape):
        # Properly initialize the global_step as a trainable weight
        self.global_step = self.add_weight(
            name='global_step', 
            shape=(), 
            initializer=tf.zeros_initializer(), 
            dtype=tf.float32, 
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Get the original dtype of the inputs
        input_dtype = inputs.dtype

        # Cast inputs to float32 for operations
        inputs = tf.cast(inputs, dtype=tf.float32)

        if training:
            # Increment global step 
            self.global_step.assign_add(1.0)

            # Calculate the current drop probability based on warmup epochs
            warmup_drop_prob = self.drop_prob_start + (self.drop_prob_end - self.drop_prob_start) * (self.global_step / self.warmup_epochs)
            
            random_tensor = 1.0 - warmup_drop_prob
            # Ensure the random tensor has the same dtype as the input (float16)
            random_tensor += tf.random.uniform([tf.shape(inputs)[0], 1, 1, 1], dtype=tf.float16)
            # Cast random_tensor back to float16 to match inputs dtype
            random_tensor = tf.cast(random_tensor, dtype=tf.float32)
            
            binary_tensor = tf.floor(random_tensor)
            output = inputs * binary_tensor
            output = tf.cast(output, dtype=tf.float16)
            output = output / (1.0 - warmup_drop_prob)

            # Return the output cast back to the original dtype (e.g., float16)
            return tf.cast(output, dtype=input_dtype)  # Cast back to the original dtype (float16)
        else:
            return inputs

    def get_config(self):
        config = super(WarmUpDropPath, self).get_config()
        config.update({
            "drop_prob_start": self.drop_prob_start, 
            "drop_prob_end": self.drop_prob_end, 
            "warmup_epochs": self.warmup_epochs
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




# Residual Block with WarmUpDropPath
def residual_block(input_tensor, filters, kernel_size=3, dropout_rate=0.3, drop_prob_start=0.0, drop_prob=0.3, warmup_epochs=10):
    # First convolution layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(input_tensor)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Second convolution layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = WarmUpDropPath(drop_prob_start=drop_prob_start, drop_prob_end=drop_prob, warmup_epochs=warmup_epochs)(x)

    # Resize input tensor to match the shape of x for addition
    if input_tensor.shape[-1] != x.shape[-1]:
        input_tensor_resized = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding="same")(input_tensor)
    else:
        input_tensor_resized = input_tensor

    # Add the input tensor to the output of the second convolution
    return tf.keras.layers.Add()([x, input_tensor_resized])

# Add GroupNormalization instead of BatchNormalization
def group_normalization(input_tensor, num_groups=32, epsilon=1e-5):
    return tfkl.GroupNormalization(groups=num_groups, axis=-1, epsilon=epsilon)(input_tensor)


# Learnable Upsampling (Deconvolution)
def learnable_upsample(input_tensor, filters=64, kernel_size=3):
    return tfkl.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same')(input_tensor)


    
def multi_scale_attention_fusion(local_output, global_output):
    # Ensure compatible shapes by using a convolution to match channels
    local_output = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, padding='same')(local_output)
    
    # Apply attention at multiple scales
    local_attention_small = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, activation='sigmoid', padding='same', name='local_attention_small')(local_output)
    global_attention_small = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, activation='sigmoid', padding='same', name='global_attention_small')(global_output)
    
    local_attention_large = tfkl.Conv2D(global_output.shape[-1], kernel_size=3, activation='sigmoid', padding='same', name='local_attention_large')(local_output)
    global_attention_large = tfkl.Conv2D(global_output.shape[-1], kernel_size=3, activation='sigmoid', padding='same', name='global_attention_large')(global_output)
    
    # Apply attention at different scales
    weighted_local_output_small = tfkl.Multiply()([local_attention_small, local_output])
    weighted_global_output_small = tfkl.Multiply()([global_attention_small, global_output])

    weighted_local_output_large = tfkl.Multiply()([local_attention_large, local_output])
    weighted_global_output_large = tfkl.Multiply()([global_attention_large, global_output])

    # Fuse features at multiple scales
    fused_small = tfkl.Add()([weighted_local_output_small, weighted_global_output_small])
    fused_large = tfkl.Add()([weighted_local_output_large, weighted_global_output_large])
    
    # Combine features from different scales
    final_fusion = tfkl.Add()([fused_small, fused_large])
    return final_fusion

def residual_attention_fusion(local_output, global_output):
    # Ensure local_output has the same number of channels as global_output
    if local_output.shape[-1] != global_output.shape[-1]:
        local_output = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, padding='same')(local_output)
    
    # Compute attention maps for both local and global outputs
    local_attention = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, activation='sigmoid', padding='same', name='local_attention')(local_output)
    global_attention = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, activation='sigmoid', padding='same', name='global_attention')(global_output)
    
    # Apply attention maps as residuals using Keras layers
    local_residual_output = tfkl.Add()([local_output, tfkl.Multiply()([local_attention, local_output])])
    global_residual_output = tfkl.Add()([global_output, tfkl.Multiply()([global_attention, global_output])])
    
    # Fuse residual outputs using Keras layer
    fused_residual_features = tfkl.Add()([local_residual_output, global_residual_output])
    
    return fused_residual_features
    
def cross_branch_attention_interaction(local_output, global_output):
    # Ensure local_output has the same number of channels as global_output
    if local_output.shape[-1] != global_output.shape[-1]:
        local_output = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, padding='same')(local_output)
    
    # Cross-attention between local and global outputs
    local_to_global_attention = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, activation='sigmoid', padding='same', name='local_to_global_attention')(local_output)
    global_to_local_attention = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, activation='sigmoid', padding='same', name='global_to_local_attention')(global_output)
    
    # Apply cross-attention to features from both branches using Keras layers
    local_interacted_output = tfkl.Multiply()([local_to_global_attention, global_output])
    global_interacted_output = tfkl.Multiply()([global_to_local_attention, local_output])
    
    # Fuse the cross-interacted features using Keras layer
    fused_cross_attention_features = tfkl.Add()([local_interacted_output, global_interacted_output])
    
    return fused_cross_attention_features
    
def final_feature_fusion(local_output, global_output):
    # Ensure local_output has the same number of channels as global_output
    if local_output.shape[-1] != global_output.shape[-1]:
        local_output = tfkl.Conv2D(global_output.shape[-1], kernel_size=1, padding='same')(local_output)
    
    # Apply multi-scale attention fusion
    multi_scale_fusion = multi_scale_attention_fusion(local_output, global_output)
    
    # Apply residual attention fusion
    residual_fusion = residual_attention_fusion(local_output, global_output)
    
    # Apply cross-branch interaction
    cross_branch_fusion = cross_branch_attention_interaction(local_output, global_output)
    
    # Combine all fused features using Keras layers
    final_fusion = tfkl.Add()([multi_scale_fusion, residual_fusion])
    final_fusion = tfkl.Add()([final_fusion, cross_branch_fusion])
    
    return final_fusion
    
def get_model(input_shape, num_classes, dropout_rate=0.3, drop_prob=0.2):
    """
    Constructs a dual-branch network with U-Net (local branch) and a global context U-Net branch.
    """

    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Local Branch (U-Net backbone)
    down_block_1 = unet_block(input_layer, 32, dropout_rate=dropout_rate, stack=3, name='down_block1_')
    down_block_1 = se_block(down_block_1, reduction_ratio=8, name='se_down_block1_')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = unet_block(d1, 64, dropout_rate=dropout_rate, stack=3, name='down_block2_')
    down_block_2 = se_block(down_block_2, reduction_ratio=8, name='se_down_block2_')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    down_block_3 = unet_block(d2, 128, dropout_rate=dropout_rate, stack=3, name='down_block3_')
    down_block_3 = se_block(down_block_3, reduction_ratio=8, name='se_down_block3_')
    d3 = tfkl.MaxPooling2D()(down_block_3)

    bottleneck = unet_block(d3, 256, dropout_rate=dropout_rate, stack=3, name='bottleneck_block_')
    bottleneck = se_block(bottleneck, reduction_ratio=8, name='se_bottleneck')

    u1 = learnable_upsample(bottleneck, 128, kernel_size=3)
    u1 = tfkl.Concatenate()([u1, down_block_3])
    u1 = residual_block(u1, 128, kernel_size=3, dropout_rate=dropout_rate, drop_prob=drop_prob)
    u1 = se_block(u1, reduction_ratio=8, name='se_up_block1_')

    u2 = learnable_upsample(u1, 64, kernel_size=3)
    u2 = tfkl.Concatenate()([u2, down_block_2])
    u2 = residual_block(u2, 64, kernel_size=3, dropout_rate=dropout_rate, drop_prob=drop_prob)
    u2 = se_block(u2, reduction_ratio=8, name='se_up_block2_')

    u3 = learnable_upsample(u2, 32, kernel_size=3)
    u3 = tfkl.Concatenate()([u3, down_block_1])
    u3 = residual_block(u3, 32, kernel_size=3, dropout_rate=dropout_rate, drop_prob=drop_prob)
    local_output = se_block(u3, reduction_ratio=8, name='se_up_block3_')

    # Global Branch (U-Net for Global Features)
    global_input = tfkl.AveragePooling2D(pool_size=8, name='global_downsample')(input_layer)  # Initial downsampling
    g_down_block_1 = unet_block(global_input, 64, dropout_rate=dropout_rate, stack=3, name='global_down_block1_')
    g1 = tfkl.MaxPooling2D()(g_down_block_1)

    g_down_block_2 = unet_block(g1, 128, dropout_rate=dropout_rate, stack=3, name='global_down_block2_')
    g2 = tfkl.MaxPooling2D()(g_down_block_2)

    g_down_block_3 = unet_block(g2, 256, dropout_rate=dropout_rate, stack=3, name='global_down_block3_')
    g3 = tfkl.MaxPooling2D()(g_down_block_3)

    global_bottleneck = unet_block(g3, 512, dropout_rate=dropout_rate, stack=3, name='global_bottleneck_block_')

    g_u1 = learnable_upsample(global_bottleneck, 256, kernel_size=3)
    g_u1 = tfkl.Concatenate()([g_u1, g_down_block_3])
    g_u1 = residual_block(g_u1, 256, kernel_size=3, dropout_rate=dropout_rate, drop_prob=drop_prob)

    g_u2 = learnable_upsample(g_u1, 128, kernel_size=3)
    g_u2 = tfkl.Concatenate()([g_u2, g_down_block_2])
    g_u2 = residual_block(g_u2, 128, kernel_size=3, dropout_rate=dropout_rate, drop_prob=drop_prob)

    g_u3 = learnable_upsample(g_u2, 64, kernel_size=3)
    g_u3 = tfkl.Concatenate()([g_u3, g_down_block_1])
    global_output = unet_block(g_u3, 64, dropout_rate=dropout_rate, stack=3, name='global_up_block3_')

    # Upsample the global branch to match local output
    global_output = tfkl.UpSampling2D(size=8, interpolation='bilinear', name='global_upsample')(global_output)

    # Feature Fusion
    fused_features = final_feature_fusion(local_output, global_output)

    fused_output = tfkl.Conv2D(
        num_classes,
        kernel_size=1,
        padding='same',
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name='fused_output_layer'
    )(fused_features)

    model = tf.keras.Model(inputs=input_layer, outputs=fused_output, name='Dual_Branch_Global_U_Net')
    return model