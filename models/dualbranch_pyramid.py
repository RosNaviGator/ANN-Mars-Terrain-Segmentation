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
    
    
def aspp_block(input_tensor, filters, name='aspp'):
    """
    Atrous Spatial Pyramid Pooling (ASPP) block with consistent spatial dimensions
    """
    # Different atrous rates for multi-scale feature extraction
    rates = [1, 3, 6, 9]
    aspp_branches = []
    
    # 1x1 Convolution
    branch1 = tfkl.Conv2D(
        filters, 
        kernel_size=1, 
        padding='same', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name=f'{name}_1x1_conv'
    )(input_tensor)
    aspp_branches.append(branch1)
    
    # Atrous Convolutions with different rates
    for rate in rates:
        branch = tfkl.Conv2D(
            filters, 
            kernel_size=3, 
            padding='same', 
            dilation_rate=rate,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
            name=f'{name}_atrous_conv_{rate}'
        )(input_tensor)
        aspp_branches.append(branch)
    
    # Global Average Pooling branch with upsampling
    gap_branch = tfkl.GlobalAveragePooling2D()(input_tensor)
    gap_branch = tfkl.Reshape((1, 1, input_tensor.shape[-1]))(gap_branch)
    gap_branch = tfkl.Conv2D(
        filters, 
        kernel_size=1, 
        padding='same', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name=f'{name}_gap_conv'
    )(gap_branch)
    # Upsample to match input spatial dimensions
    gap_branch = tfkl.UpSampling2D(size=(input_tensor.shape[1], input_tensor.shape[2]), interpolation='bilinear')(gap_branch)
    aspp_branches.append(gap_branch)
    
    # Concatenate and reduce
    aspp_output = tfkl.Concatenate(name=f'{name}_concat')(aspp_branches)
    aspp_output = tfkl.Conv2D(
        filters, 
        kernel_size=1, 
        padding='same', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
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
    
    

def get_enhanced_global_branch(input_layer, dropout_rate=0.3):
    """
    Advanced Global Branch with Multi-Scale Context Aggregation
    Focuses on extracting comprehensive global context with high-performance techniques
    """
    # Initial aggressive downsampling for global context
    global_input = tfkl.AveragePooling2D(pool_size=4, name='global_context_downsample')(input_layer)

    # Multi-Scale Feature Extraction with Enhanced ASPP
    aspp_features = aspp_block(global_input, 256, name='global_enhanced_aspp')

    # Pyramid Pooling Module for Comprehensive Context
    pyramid_features = []
    pool_sizes = [1, 2, 4, 8]
    for pool_size in pool_sizes:
        pyramid_branch = tfkl.AveragePooling2D(pool_size=pool_size)(global_input)
        pyramid_branch = tfkl.Conv2D(
            64, 
            kernel_size=1, 
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
            name=f'pyramid_pool_{pool_size}'
        )(pyramid_branch)
        pyramid_branch = tfkl.UpSampling2D(size=pool_size, interpolation='bilinear')(pyramid_branch)
        pyramid_features.append(pyramid_branch)

    # Concatenate Pyramid Features
    pyramid_context = tfkl.Concatenate(name='pyramid_context_concat')(pyramid_features + [aspp_features])

    # Context Refinement Block
    context_refined = tfkl.Conv2D(
        256, 
        kernel_size=3, 
        padding='same', 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name='context_refinement'
    )(pyramid_context)

    # Channel Attention Refinement
    channel_attention = tfkl.GlobalAveragePooling2D()(context_refined)
    channel_attention = tfkl.Dense(
        256 // 4, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name='channel_attention_1'
    )(channel_attention)
    channel_attention = tfkl.Dense(
        256, 
        activation='sigmoid', 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        name='channel_attention_2'
    )(channel_attention)
    channel_attention = tfkl.Reshape((1, 1, 256))(channel_attention)
    
    # Apply Channel Attention
    context_refined = tfkl.Multiply(name='channel_attentive_context')([context_refined, channel_attention])

    # Final Upsampling to Original Resolution
    global_output = tfkl.UpSampling2D(
        size=4, 
        interpolation='bilinear', 
        name='global_context_upsample'
    )(context_refined)

    # Add Stochastic Depth for Regularization
    global_output = stochastic_depth_layer(
        global_output, 
        survival_probability=0.8, 
        name='global_stochastic_depth'
    )

    return global_output

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

    model = tf.keras.Model(inputs=input_layer, outputs=fused_output, name='Dual_Branch_pyramid')
    return model