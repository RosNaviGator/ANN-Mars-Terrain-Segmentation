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
            kernel_initializer='he_normal',  # Added kernel_initializer
            name=name + 'conv' + str(i + 1)
        )(x)
        x = tfkl.BatchNormalization(name=name + 'bn' + str(i + 1))(x)
        x = tfkl.Activation(activation, name=name + 'activation' + str(i + 1))(x)
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate, name=name + 'dropout' + str(i + 1))(x)
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
        kernel_initializer='glorot_uniform',  # Added kernel_initializer
        name=name + 'dense1'
    )(se)
    se = tfkl.Dense(
        channels,
        activation='sigmoid',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        kernel_initializer='glorot_uniform',  # Added kernel_initializer
        name=name + 'dense2'
    )(se)
    se = tfkl.Multiply(name=name + 'multiply')([input_tensor, se])
    return se

    
def cross_scale_attention_fusion(local_output, global_output, name='csa_fusion'):
    """
    Implements Cross-Scale Attention for feature fusion
    """
    concat_features = tfkl.Concatenate(name=f'{name}_concat')([local_output, global_output])
    channels = concat_features.shape[-1]
    
    gap = tfkl.GlobalAveragePooling2D(name=f'{name}_gap')(concat_features)
    attention_weights = tfkl.Dense(
        channels, 
        activation='sigmoid', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        kernel_initializer='glorot_uniform',  # Added kernel_initializer
        name=f'{name}_attention_weights'
    )(gap)
    attention_weights = tfkl.Reshape((1, 1, channels))(attention_weights)
    attended_features = tfkl.Multiply(name=f'{name}_attended_features')([concat_features, attention_weights])
    fused_output = tfkl.Conv2D(
        local_output.shape[-1],
        kernel_size=1,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        kernel_initializer='he_normal',  # Added kernel_initializer
        name=f'{name}_fusion_conv'
    )(attended_features)
    
    return fused_output

    
def aspp_block(input_tensor, filters, name='aspp'):
    """
    Atrous Spatial Pyramid Pooling (ASPP) block with consistent spatial dimensions
    """
    rates = [1, 3, 6, 9]
    aspp_branches = []
    
    branch1 = tfkl.Conv2D(
        filters, 
        kernel_size=1, 
        padding='same', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        kernel_initializer='he_normal',  # Added kernel_initializer
        name=f'{name}_1x1_conv'
    )(input_tensor)
    aspp_branches.append(branch1)
    
    for rate in rates:
        branch = tfkl.Conv2D(
            filters, 
            kernel_size=3, 
            padding='same', 
            dilation_rate=rate,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
            kernel_initializer='he_normal',  # Added kernel_initializer
            name=f'{name}_atrous_conv_{rate}'
        )(input_tensor)
        aspp_branches.append(branch)
    
    gap_branch = tfkl.GlobalAveragePooling2D()(input_tensor)
    gap_branch = tfkl.Reshape((1, 1, input_tensor.shape[-1]))(gap_branch)
    gap_branch = tfkl.Conv2D(
        filters, 
        kernel_size=1, 
        padding='same', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        kernel_initializer='he_normal',  # Added kernel_initializer
        name=f'{name}_gap_conv'
    )(gap_branch)
    gap_branch = tfkl.UpSampling2D(size=(input_tensor.shape[1], input_tensor.shape[2]), interpolation='bilinear')(gap_branch)
    aspp_branches.append(gap_branch)
    
    aspp_output = tfkl.Concatenate(name=f'{name}_concat')(aspp_branches)
    aspp_output = tfkl.Conv2D(
        filters, 
        kernel_size=1, 
        padding='same', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        kernel_initializer='he_normal',  # Added kernel_initializer
        name=f'{name}_reduce_conv'
    )(aspp_output)
    
    return aspp_output

    
@tf.keras.utils.register_keras_serializable()
class StochasticDepthLayer(tf.keras.layers.Layer):
    def __init__(self, survival_probability=0.8, **kwargs):
        """
        Custom Stochastic Depth Layer for TensorFlow/Keras.
        
        Args:
            survival_probability (float): Probability of layer survival during training.
        """
        super(StochasticDepthLayer, self).__init__(**kwargs)
        self.survival_probability = survival_probability

    @tf.function
    def call(self, inputs, training=None):
        """
        Apply stochastic depth during training.
        
        Args:
            inputs (tf.Tensor): Input tensor
            training (bool, optional): Training flag
        
        Returns:
            tf.Tensor: Processed tensor with stochastic depth applied
        """
        # Determine training status
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        # If not training, return inputs as-is
        if not training:
            return inputs
        
        # Generate random tensor for stochastic depth
        batch_size = tf.shape(inputs)[0]
        random_tensor = self.survival_probability + tf.random.uniform(
            [batch_size, 1, 1, 1], 
            dtype=inputs.dtype
        )
        
        # Create binary mask
        binary_tensor = tf.floor(random_tensor)
        
        # Scale and apply binary mask
        return (inputs / self.survival_probability) * binary_tensor

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape (tf.TensorShape): Input tensor shape
        
        Returns:
            tf.TensorShape: Output tensor shape
        """
        return input_shape

    def get_config(self):
        """
        Enable layer configuration serialization.
        
        Returns:
            dict: Layer configuration
        """
        config = super(StochasticDepthLayer, self).get_config()
        config.update({
            'survival_probability': self.survival_probability
        })
        return config

def stochastic_depth_layer(inputs, survival_probability=0.8, name='stochastic_depth'):
    """
    Convenience function to apply stochastic depth.
    
    Args:
        inputs (tf.Tensor): Input tensor
        survival_probability (float): Probability of layer survival
        name (str): Layer name
    
    Returns:
        tf.Tensor: Tensor with stochastic depth applied
    """
    return StochasticDepthLayer(
        survival_probability=survival_probability, 
        name=name
    )(inputs)
    
def transformer_feature_extractor(input_tensor, num_transformer_blocks=2, embed_dim=128):
    # Project input to embedding dimension
    x = tfkl.Conv2D(embed_dim, kernel_size=1, padding='same')(input_tensor)
    
    # Split into patches or tokens
    batch_size, height, width, _ = x.shape
    x = tfkl.Reshape((height * width, embed_dim))(x)

    # Simplified transformer blocks
    for _ in range(num_transformer_blocks):
        # Lightweight self-attention
        attention = tfkl.Dense(embed_dim)(x)
        x = x + attention  # Simple residual connection

        # Lightweight feed-forward
        ff = tfkl.Dense(embed_dim, activation='gelu')(x)
        x = x + ff

    # Reshape back to spatial format
    x = tfkl.Reshape((height, width, embed_dim))(x)

    return x
    
def get_original_global_branch(input_layer, dropout_rate=0.3):
    """
    Enhanced Global Branch with ASPP, Skip Connections, and Stochastic Depth
    """
    # Initial downsampling with skip connection preservation
    global_input = tfkl.AveragePooling2D(pool_size=4, name='global_downsample')(input_layer)
    
    # Encoder with skip connection storage
    skip_connections = []
    
    # First encoder block with skip connection
    global_features_1 = unet_block(global_input, 64, dropout_rate=dropout_rate, stack=2, name='global_encoder_1_')
    skip_connections.append(global_features_1)
    global_features_1 = tfkl.MaxPooling2D()(global_features_1)
    
    # Second encoder block with skip connection
    global_features_2 = unet_block(global_features_1, 128, dropout_rate=dropout_rate, stack=2, name='global_encoder_2_')
    skip_connections.append(global_features_2)
    global_features_2 = tfkl.MaxPooling2D()(global_features_2)
    
    # ASPP for enhanced global context
    global_features = aspp_block(global_features_2, 256, name='global_aspp')
    
    # Stochastic depth regularization
    global_features = stochastic_depth_layer(global_features, survival_probability=0.8)
    
    # Decoder with skip connections
    u1 = tfkl.UpSampling2D()(global_features)
    u1 = tfkl.Concatenate()([u1, skip_connections[1]])
    u1 = unet_block(u1, 128, dropout_rate=dropout_rate, stack=2, name='global_decoder_1_')
    
    u2 = tfkl.UpSampling2D()(u1)
    u2 = tfkl.Concatenate()([u2, skip_connections[0]])
    u2 = unet_block(u2, 64, dropout_rate=dropout_rate, stack=2, name='global_decoder_2_')
    
    # Final upsampling to original input resolution
    global_output = tfkl.UpSampling2D(size=4, interpolation='bilinear', name='global_upsample')(u2)
    
    return global_output
    
def get_enhanced_global_branch(input_layer, dropout_rate=0.3):
    # Existing global branch logic
    original_global_output = get_original_global_branch(input_layer, dropout_rate=0.3)
    
    # Transformer feature extraction
    transformer_global_features = transformer_feature_extractor(input_layer)
    
    # Fuse within global branch
    fused_global_output = cross_scale_attention_fusion(
        original_global_output, 
        transformer_global_features
    )
    
    return fused_global_output


def get_model(input_shape, num_classes, dropout_rate=0.3):
    """
    Constructs a dual-branch network with U-Net (local branch) and a global context branch.
    """
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # Local Branch (U-Net backbone)
    down_block_1 = unet_block(input_layer, 32, dropout_rate=dropout_rate, stack=2, name='down_block1_')
    down_block_1 = se_block(down_block_1, reduction_ratio=8, name='se_down_block1_')
    d1 = tfkl.MaxPooling2D()(down_block_1)

    down_block_2 = unet_block(d1, 64, dropout_rate=dropout_rate, stack=2, name='down_block2_')
    down_block_2 = se_block(down_block_2, reduction_ratio=8, name='se_down_block2_')
    d2 = tfkl.MaxPooling2D()(down_block_2)

    down_block_3 = unet_block(d2, 128, dropout_rate=dropout_rate, stack=2, name='down_block3_')
    down_block_3 = se_block(down_block_3, reduction_ratio=8, name='se_down_block3_')
    d3 = tfkl.MaxPooling2D()(down_block_3)

    bottleneck = unet_block(d3, 256, dropout_rate=dropout_rate, stack=2, name='bottleneck_block_')
    bottleneck = se_block(bottleneck, reduction_ratio=8, name='se_bottleneck')

    u1 = tfkl.UpSampling2D()(bottleneck)
    u1 = tfkl.Concatenate()([u1, down_block_3])
    u1 = unet_block(u1, 128, dropout_rate=dropout_rate, stack=2, name='up_block1_')
    u1 = se_block(u1, reduction_ratio=8, name='se_up_block1_')

    u2 = tfkl.UpSampling2D()(u1)
    u2 = tfkl.Concatenate()([u2, down_block_2])
    u2 = unet_block(u2, 64, dropout_rate=dropout_rate, stack=2, name='up_block2_')
    u2 = se_block(u2, reduction_ratio=8, name='se_up_block2_')

    u3 = tfkl.UpSampling2D()(u2)
    u3 = tfkl.Concatenate()([u3, down_block_1])
    u3 = unet_block(u3, 32, dropout_rate=dropout_rate, stack=2, name='up_block3_')
    local_output = se_block(u3, reduction_ratio=8, name='se_up_block3_')

    # Global Branch
    global_output = get_enhanced_global_branch(input_layer, dropout_rate=dropout_rate)

    # Feature Fusion
    fused_features = cross_scale_attention_fusion(local_output, global_output, name='feature_fusion')
    fused_output = tfkl.Conv2D(
        num_classes,
        kernel_size=1,
        padding='same',
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name='fused_output_layer'
    )(fused_features)

    model = tf.keras.Model(inputs=input_layer, outputs=fused_output, name='Dual_Branch')
    return model