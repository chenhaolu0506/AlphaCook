# import math
# import numpy as np
# import tensorflow as tf


# class AttentionMatrix(tf.keras.layers.Layer):

#     def __init__(self, *args, use_mask=False, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.use_mask = use_mask

#     def call(self, inputs):
#         """
#         STUDENT MUST WRITE:

#         Computes attention given key and query matrices.

#         :param K: is [batch_size x window_size_keys x embedding_size]
#         :param Q: is [batch_size x window_size_queries x embedding_size]
#         :return: attention matrix
#         """
#         K, Q = inputs
#         window_size_queries = Q.get_shape()[1]  # window size of queries
#         window_size_keys    = K.get_shape()[1]  # window size of keys

#         ## Fill triangle below diagonal of matrix with negative infinity and top part with 0.
#         ## This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal. 
#         ## Tile this upward to be compatible with addition against computed attention scores.
#         mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
#         mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
#         atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

#         # TODO:
#         # 1) compute attention weights using queries and key matrices 
#         #       - if use_mask==True, then make sure to add the attention mask before softmax
#         # 2) return the attention matrix

#         # Check lecture slides for how to compute self-attention
#         # Remember:
#         # - Q is [batch_size x window_size_queries x embedding_size]
#         # - K is [batch_size x window_size_keys x embedding_size]
#         # - Mask is [batch_size x window_size_queries x window_size_keys]

#         # Here, queries are matrix multiplied with the transpose of keys to produce for every query vector, weights per key vector.
#         # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
#         # Those weights are then used to create linear combinations of the corresponding values for each query.
#         # Those queries will become the new embeddings. Return attention score as per lecture slides.
#         atten_score = tf.matmul(Q, K, transpose_b=True) / math.sqrt(window_size_keys)
#         if self.use_mask:
#             atten_score += atten_mask
        
#         self.scores = tf.nn.softmax(atten_score)

#         return self.scores


# class AttentionHead(tf.keras.layers.Layer):
#     def __init__(self, input_size, output_size, is_self_attention, **kwargs):
#         super(AttentionHead, self).__init__(**kwargs)
#         self.use_mask = is_self_attention

#         # TODO:
#         # Initialize the weight matrices for K, V, and Q.
#         # They should be able to multiply an input_size vector to produce an output_size vector
#         self.weigt_intitializer = tf.keras.initializers.GlorotNormal()

#         self.k_weight = self.add_weight("K", shape=(input_size, output_size))
#         self.q_weight = self.add_weight("Q", shape=(input_size, output_size))
#         self.v_weight = self.add_weight("V", shape=(input_size, output_size))

#         self.K = self.Q = self.V = 0

#         self.attentionMatrix = AttentionMatrix(use_mask=self.use_mask)

#     @tf.function
#     def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
#         """
#         STUDENT MUST WRITE:

#         This functions runs a single attention head.

#         :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
#         :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
#         :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
#         :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
#         """

#         # TODO:
#         # - Apply 3 matrix products to turn inputs into keys, values, and queries. 
#         # - You will need to use tf.tensordot for this.
#         # - Call your AttentionMatrix layer with the keys and queries.
#         # - Apply the attention matrix to the values.

#         self.K = tf.matmul(inputs_for_keys, self.k_weight)
#         self.V = tf.matmul(inputs_for_values, self.v_weight)
#         self.Q = tf.matmul(inputs_for_queries, self.q_weight)

#         atten_scores = self.attentionMatrix.call((self.K, self.Q))

#         return tf.matmul(atten_scores, self.V)


# class MultiHeadedAttention(tf.keras.layers.Layer):
#     def __init__(self, emb_sz, use_mask, **kwargs):
#         super(MultiHeadedAttention, self).__init__(**kwargs)

#         self.use_mask = use_mask

#         ## TODO: Add 3 heads as appropriate and any other necessary components
#         self.weigt_intitializer = tf.keras.initializers.GlorotNormal()

#         self.head1 = AttentionHead(emb_sz, emb_sz//3, use_mask)
#         self.head2 = AttentionHead(emb_sz, emb_sz//3, use_mask)
#         self.head3 = AttentionHead(emb_sz, emb_sz//3, use_mask)

#         self.linear = tf.keras.layers.Dense(emb_sz)

#     @tf.function
#     def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
#         """
#         TODO: FOR CS2470 STUDENTS:

#         This functions runs a multiheaded attention layer.

#         Requirements:
#             - Splits data for 3 different heads of size embed_sz/3
#             - Create three different attention heads
#             - Concatenate the outputs of these heads together
#             - Apply a linear layer

#         :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
#         :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
#         :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
#         :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
#         """

#         z1 = self.head1(inputs_for_keys, inputs_for_values, inputs_for_queries)
#         z2 = self.head2(inputs_for_keys, inputs_for_values, inputs_for_queries)
#         z3 = self.head3(inputs_for_keys, inputs_for_values, inputs_for_queries)

#         z = tf.concat([z1, z2, z3], axis=-1)
#         return self.linear(z)


# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, emb_sz, isMultiHeadedAttention=False, **kwargs):
#         super(TransformerBlock, self).__init__(**kwargs)

#         # TODO:
#         # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
#         # 2) For 2470 students, use multiheaded attention

#         self.ff_layer = tf.keras.layers.Dense(emb_sz, activation="relu")

#         self.self_atten         = AttentionHead(emb_sz, emb_sz, True)  if not isMultiHeadedAttention else MultiHeadedAttention(emb_sz, True)
#         self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) if not isMultiHeadedAttention else MultiHeadedAttention(emb_sz, False)
#         self.layer_norm = tf.keras.layers.LayerNormalization()
#         self.add = tf.keras.layers.Add()
#         self.dropout = tf.keras.layers.Dropout(0.5)

#     @tf.function
#     def call(self, inputs, context_sequence):
#         """
#         This functions calls a transformer block.

#         TODO:
#         1) compute MASKED attention on the inputs
#         2) residual connection and layer normalization
#         3) computed UNMASKED attention using context
#         4) residual connection and layer normalization
#         5) feed forward layer
#         6) residual layer and layer normalization
#         7) return relu of tensor

#         NOTES: This article may be of great use:
#         https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

#         :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
#         :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
#         :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
#         """
#         x = inputs

#         self_atten = self.self_atten(x, x, x)
#         x = self.add([self_atten, x])
#         x = self.dropout(self.layer_norm(x))
        
#         context_atten = self.self_context_atten(context_sequence, context_sequence, x)
#         x = self.add([context_atten, x])
#         x = self.dropout(self.layer_norm(x))
        
#         x = self.ff_layer(x)
#         x = self.dropout(self.layer_norm(x))

#         return tf.nn.relu(x)


# def positional_encoding(length, depth):
#     ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
#     depth = depth/2
#     ## Generate a range of positions and depths 
#     positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
#     depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
#     ## Compute range of radians to take the sine and cosine of.
#     angle_rates = 1 / (10000**depths)               # (1, depth)
#     angle_rads = positions * angle_rates            # (pos, depth)
#     pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
#     ## This serves as offset for the Positional Encoding
#     return tf.cast(pos_encoding, dtype=tf.float32)


# class PositionalEncoding(tf.keras.layers.Layer):
#     """
#     STUDENT MUST WRITE:

#     Embed labels and apply positional offsetting
#     """
#     def __init__(self, vocab_size, embed_size, window_size):
#         super().__init__()
#         self.embed_size = embed_size
#         self.vocab_size = vocab_size
#         self.window_size = window_size

#         ## TODO: Implement Components

#         ## Embed labels into an optimizable embedding space
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True) 

#         ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
#         ## HINT: May want to use the function above...
#         self.pos_encoding = positional_encoding(length=2048, depth=embed_size)

#     def call(self, x):
#         ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.

#         length = tf.shape(x)[1]
#         x = self.embedding(x)
#         # This factor sets the relative scale of the embedding and positonal_encoding.
#         x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
#         x = x + self.pos_encoding[tf.newaxis, :length, :]
#         return x


import numpy as np
import tensorflow as tf


class AttentionMatrix(tf.keras.layers.Layer):
    
    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask
    
    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys = K.get_shape()[1]  # window size of keys
        
        ## Fill triangle below diagonal of matrix with negative infinity and top part with 0.
        ## This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal. 
        ## Tile this upward to be compatible with addition against computed attention scores.
        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]),
                             [tf.shape(input=K)[0], 1, 1])
        
        # TODO:
        # 1) compute attention weights using queries and key matrices 
        #       - if use_mask==True, then make sure to add the attention mask before softmax
        # 2) return the attention matrix
        
        # Check lecture slides for how to compute self-attention
        # Remember:
        # - Q is [batch_size x window_size_queries x embedding_size]
        # - K is [batch_size x window_size_keys x embedding_size]
        # - Mask is [batch_size x window_size_queries x window_size_keys]
        
        # Here, queries are matrix multiplied with the transpose of keys to produce for every query vector, weights per key vector.
        # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
        # Those weights are then used to create linear combinations of the corresponding values for each query.
        # Those queries will become the new embeddings. Return attention score as per lecture slides.
        # print("Q shape: ", Q.shape)
        # print("K shape: ", K.shape)
        atten_weights = tf.matmul(Q, tf.transpose(a=K, perm=[0, 2, 1])) / np.sqrt(K.shape[1])
        if self.use_mask:
            atten_weights += atten_mask
        atten_weights = tf.nn.softmax(atten_weights, axis=-1)
        return atten_weights


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention
        
        # TODO:
        # Initialize the weight matrices for K, V, and Q. Need to use tensordot later.
        # They should be able to multiply an input_size vector to produce an output_size vector
        self.K = tf.keras.layers.Dense(output_size, input_shape=(input_size,), use_bias=False)
        self.V = tf.keras.layers.Dense(output_size, input_shape=(input_size,), use_bias=False)
        self.Q = tf.keras.layers.Dense(output_size, input_shape=(input_size,), use_bias=False)
        self.attention_matrix = AttentionMatrix(use_mask=self.use_mask)
    
    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        
        # TODO:
        # - Apply 3 matrix products to turn inputs into keys, values, and queries. 
        # - You will need to use tf.tensordot for this.
        # - Call your AttentionMatrix layer with the keys and queries.
        # - Apply the attention matrix to the values.
        
        K = self.K(inputs_for_keys)
        V = self.V(inputs_for_values)
        Q = self.Q(inputs_for_queries)
        # print("K shape: ", K.shape)
        # print("V shape: ", V.shape)
        # print("Q shape: ", Q.shape)
        
        return tf.matmul(self.attention_matrix([K, Q]), V)


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        
        ## TODO: Add 3 heads as appropriate and any other necessary components
        # This class should create 3 self attention heads of input size batch_sz x window_sz x emb_sz
        # print('emb_sz: ', emb_sz)
        self.head1 = AttentionHead(emb_sz, emb_sz // 3, use_mask)
        self.head2 = AttentionHead(emb_sz, emb_sz // 3, use_mask)
        self.head3 = AttentionHead(emb_sz, emb_sz // 3, use_mask)
        self.linear = tf.keras.layers.Dense(emb_sz)
    
    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        TODO: FOR CS2470 STUDENTS:

        This functions runs a multiheaded attention layer.

        Requirements:
            - Splits data for 3 different heads of size embed_sz/3
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        head1 = self.head1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        head2 = self.head2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        head3 = self.head3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        # concatenate the outputs of these heads together
        concat = tf.concat([head1, head2, head3], axis=-1)
        # apply a linear layer
        return self.linear(concat)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multi_head=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        # TODO:
        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) For 2470 students, use multiheaded attention
        
        self.ff_layer = tf.keras.layers.Dense(emb_sz, activation='relu')
        
        self.self_atten = AttentionHead(emb_sz, emb_sz, True) if not multi_head else MultiHeadedAttention(
            emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz,
                                                False) if not multi_head else MultiHeadedAttention(emb_sz,
                                                                                                   False)
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        TODO:
        1) compute MASKED attention on the inputs
        2) residual connection and layer normalization
        3) computed UNMASKED attention using context
        4) residual connection and layer normalization
        5) feed forward layer
        6) residual layer and layer normalization
        7) return relu of tensor

        NOTES: This article may be of great use:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        masked_atten = self.self_atten(inputs, inputs, inputs)
        masked_atten = self.layer_norm(masked_atten + inputs)  # residual connection and layer normalization
        unmasked_atten = self.self_context_atten(context_sequence, context_sequence, masked_atten)
        unmasked_atten = self.layer_norm(unmasked_atten + masked_atten)  # residual connection and layer normalization
        ff = self.ff_layer(unmasked_atten)
        ff = self.layer_norm(ff + unmasked_atten)  # residual connection and layer normalization
        return tf.nn.relu(ff)


def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    ## TODO: Can remove signature
    depth = depth / 2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        
        ## TODO: Implement Component
        
        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        
        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size)
    
    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        embeddings = self.embedding(x)
        embeddings *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        embeddings += self.pos_encoding
        return embeddings