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
    
@tf.keras.utils.register_keras_serializable()
class CheckpointedUNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate, stack, name=""):
        super(CheckpointedUNetBlock, self).__init__(name=name)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.stack = stack
        self.name = name
        
        # Define the layers once in __init__
        self.convs = []
        for i in range(stack):
            conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=3,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
                name=name + f'conv{i + 1}'
            )
            self.convs.append(conv)
        
        self.bn = tf.keras.layers.BatchNormalization(name=name + 'bn')
        self.activation = tf.keras.layers.Activation('relu', name=name + 'activation')
        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, name=name + 'dropout')
        else:
            self.dropout = None

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
            x = self.bn(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        return x

    def get_config(self):
        # Serialize the layer's parameters
        config = super(CheckpointedUNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'dropout_rate': self.dropout_rate,
            'stack': self.stack,
            'name': self.name
        })
        return config

        
@tf.keras.utils.register_keras_serializable()
class CheckpointedSEBlock(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, name=""):
        super(CheckpointedSEBlock, self).__init__(name=name)
        self.reduction_ratio = reduction_ratio
        self.name = name
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
        # Calculate the number of channels in the input tensor (last dimension)
        channels = input_shape[-1]
        
        # Create Dense layers using the determined number of channels
        self.dense1 = tfkl.Dense(
            channels // self.reduction_ratio,
            activation='relu',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
            name=self.name + 'dense1'
        )
        
        self.dense2 = tfkl.Dense(
            channels,
            activation='sigmoid',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
            name=self.name + 'dense2'
        )

        # Call build on the Dense layers to initialize their state
        super().build(input_shape)

    def call(self, inputs):
        # Squeeze-and-Excitation block implementation
        se = tfkl.GlobalAveragePooling2D()(inputs)
        se = tfkl.Reshape((1, 1, inputs.shape[-1]))(se)
        se = self.dense1(se)  # Apply first Dense layer
        se = self.dense2(se)  # Apply second Dense layer
        se = tfkl.Multiply(name=self.name + 'multiply')([inputs, se])  # Apply element-wise multiplication
        return se

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'name': self.name
        })
        return config

@tf.keras.utils.register_keras_serializable()
class GradientCheckpointedLayer(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        super(GradientCheckpointedLayer, self).__init__(**kwargs)
        self.layer = layer

    def call(self, inputs):
        # Use tf.recompute_grad on the layer to enable gradient checkpointing
        return tf.recompute_grad(self.layer)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'layer': self.layer})
        return config
        
@tf.keras.utils.register_keras_serializable()
class GradientCheckpointedSELayer(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        super(GradientCheckpointedSELayer, self).__init__(**kwargs)
        self.layer = layer

    def call(self, inputs):
        # Apply gradient checkpointing to SE block
        return tf.recompute_grad(self.layer)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'layer': self.layer})
        return config
    
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
class TransformerGlobalBranch(tfkl.Layer):
    def __init__(
        self, 
        num_transformer_blocks=4, 
        embed_dim=64, 
        num_heads=4, 
        mlp_ratio=2, 
        dropout_rate=0.1, 
        name="transformer_global",
        **kwargs
    ):
        super(TransformerGlobalBranch, self).__init__(name=name, **kwargs)
        self.num_transformer_blocks = num_transformer_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate

        # Layers
        self.embedding_projection = tfkl.Conv2D(embed_dim, kernel_size=1, padding="same", name=f"{name}_embedding_projection")
        self.final_projection = tfkl.Conv2D(embed_dim, kernel_size=1, padding="same", name=f"{name}_final_projection")

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.flatten = tfkl.Reshape((-1, self.embed_dim), name=f"{self.name}_flatten")
        
        self.position_embedding_layer = tfkl.Embedding(input_dim=height * width, output_dim=self.embed_dim, name=f"{self.name}_position_embedding")
        
        self.attention_layers = []
        self.norm_layers_1 = []
        self.mlp_dense1_layers = []
        self.mlp_dropout1_layers = []
        self.mlp_dense2_layers = []
        self.mlp_dropout2_layers = []
        self.norm_layers_2 = []

        for i in range(self.num_transformer_blocks):
            self.attention_layers.append(
                LocalSelfAttention(
                    chunk_size=64,  # Reduce attention computation
                    num_heads=self.num_heads, 
                    key_dim=self.embed_dim // self.num_heads, 
                    dropout=self.dropout_rate, 
                    name=f"{self.name}_local_mha_block_{i}"
                )
            )
            self.norm_layers_1.append(tfkl.LayerNormalization(name=f"{self.name}_ln_1_block_{i}"))
            self.mlp_dense1_layers.append(tfkl.Dense(self.embed_dim * self.mlp_ratio, activation="gelu", name=f"{self.name}_mlp_dense1_block_{i}"))
            self.mlp_dropout1_layers.append(tfkl.Dropout(self.dropout_rate, name=f"{self.name}_mlp_dropout1_block_{i}"))
            self.mlp_dense2_layers.append(tfkl.Dense(self.embed_dim, name=f"{self.name}_mlp_dense2_block_{i}"))
            self.mlp_dropout2_layers.append(tfkl.Dropout(self.dropout_rate, name=f"{self.name}_mlp_dropout2_block_{i}"))
            self.norm_layers_2.append(tfkl.LayerNormalization(name=f"{self.name}_ln_2_block_{i}"))
        
        super(TransformerGlobalBranch, self).build(input_shape)

    def call(self, inputs, training=None):
        # Initial projection
        x = self.embedding_projection(inputs)
        
        # Flatten the input
        batch_size = tf.shape(x)[0]
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        x = self.flatten(x)
        
        
        positions = tf.range(start=0, limit=height * width, delta=1)  # Range for all positions up to max_positions
        pos_embeddings = self.position_embedding_layer(positions)
        
        # Reshape the positional embeddings to match the flattened input tensor
        # Ensure it matches the flattened input size, i.e., (batch_size, height * width, embed_dim)
        #pos_embeddings = tf.reshape(pos_embeddings, (1, height * width, self.embed_dim))  # Shape: (1, limited_height, embed_dim)
        
        # Adjust broadcasting to match the batch size and height * width
        #pos_embeddings = tf.broadcast_to(pos_embeddings, [batch_size, height * width, self.embed_dim])  # Broadcast to match batch size and spatial size
            
        # Add positional embeddings to the input tensor
        x = x + pos_embeddings  # Add positional embeddings to features

        # Apply gradient checkpointing to each Transformer block
        for i in range(self.num_transformer_blocks):
            # Multi-Head Attention Block
            def mha_block(x):
                attn_output = self.attention_layers[i](x, x, x, training=training)
                attn_output = self.mlp_dropout1_layers[i](attn_output, training=training)
                return self.norm_layers_1[i](x + attn_output, training=training)
            
            # MLP Block
            def mlp_block(x):
                mlp_output = self.mlp_dense1_layers[i](x, training=training)
                mlp_output = self.mlp_dense2_layers[i](mlp_output, training=training)
                mlp_output = self.mlp_dropout2_layers[i](mlp_output, training=training)
                return self.norm_layers_2[i](x + mlp_output, training=training)
            
            # Apply recompute_grad to individual blocks
            x = tf.recompute_grad(mha_block)(x)
            x = tf.recompute_grad(mlp_block)(x)

        # Final projection and reshape back to the spatial dimensions
        x = self.final_projection(tf.reshape(x, (-1, tf.shape(inputs)[1], tf.shape(inputs)[2], self.embed_dim)))

        return x

    def get_config(self):
        config = super(TransformerGlobalBranch, self).get_config()
        config.update({
            "num_transformer_blocks": self.num_transformer_blocks,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
        })
        return config
        
        
@tf.keras.utils.register_keras_serializable()
class LocalSelfAttention(tfkl.MultiHeadAttention):
    def __init__(self, chunk_size=64, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None):
        # If key is not provided, use value
        if key is None:
            key = value
        
        # Break input into chunks to reduce memory usage
        batch_size, seq_len, _ = query.shape
        
        # Process in chunks to reduce memory
        outputs = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_query = query[:, start:end, :]
            chunk_key = key[:, start:end, :]
            chunk_value = value[:, start:end, :]
            
            # Apply standard MultiHeadAttention to chunks
            chunk_output = super().call(
                chunk_query, 
                chunk_value, 
                chunk_key, 
                attention_mask=attention_mask, 
                return_attention_scores=return_attention_scores,
                training=training
            )
            outputs.append(chunk_output)
        
        # Concatenate chunk results
        if return_attention_scores:
            # If attention scores are returned, handle tuple unpacking
            chunk_outputs, chunk_attention_scores = zip(*outputs)
            return tf.concat(chunk_outputs, axis=1), tf.concat(chunk_attention_scores, axis=1)
        else:
            return tf.concat(outputs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "chunk_size": self.chunk_size
        })
        return config

def get_transformer_global_branch(input_tensor, **kwargs):
    return TransformerGlobalBranch(**kwargs)(input_tensor)


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

    #print("shape of d3", d3.shape)
    # Apply gradient checkpointing to the bottleneck
    bottleneck_block = CheckpointedUNetBlock(256, dropout_rate, stack=2, name='bottleneck_block_')
    
    #bottleneck_with_checkpointing = GradientCheckpointedLayer(bottleneck_block)(d3)
    bottleneck_without_checkpointing = bottleneck_block(d3)

    #print("shape of bottleneck_with_checkpointing", bottleneck_with_checkpointing.shape)
    se_bottleneck_block = CheckpointedSEBlock(reduction_ratio=8, name='se_bottleneck')
    
    #se_bottleneck_with_checkpointing = GradientCheckpointedSELayer(se_bottleneck_block)(bottleneck_with_checkpointing)
    se_bottleneck_without_checkpointing = se_bottleneck_block(bottleneck_without_checkpointing)
    #print("shape of se_bottleneck_with_checkpointing", se_bottleneck_with_checkpointing.shape)

    u1 = tfkl.UpSampling2D()(se_bottleneck_without_checkpointing)
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
    global_output = get_transformer_global_branch(input_layer, dropout_rate=dropout_rate)

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

    model = tf.keras.Model(inputs=input_layer, outputs=fused_output, name='Dual_Branch_transformer_global')
    return model