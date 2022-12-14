# import tensorflow as tf

# from transformer import TransformerBlock, PositionalEncoding
# from attention import EncoderLayer

# ########################################################################################

# class RNNDecoder(tf.keras.layers.Layer):

#     def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

#         super().__init__(**kwargs)
#         self.vocab_size  = vocab_size
#         self.hidden_size = hidden_size
#         self.window_size = window_size

#         # TODO:
#         # Now we will define image and word embedding, decoder, and classification layers

#         # Define feed forward layer to embed image features into a vector 
#         # with the models hidden size
#         self.image_embedding = tf.keras.layers.Dense(self.hidden_size, activation='relu')
# #         for weight in self.image_embedding.weights:
# #             self.trainable_variables += [weight]
        
#         # Define english embedding layer:
#         self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size)
# #         for weight in self.embedding.weights:
# #             self.trainable_variables += [weight]

#         # Define decoder layer that handles language and image context:     
#         self.decoder = tf.keras.layers.GRU(self.hidden_size, return_sequences=True, return_state=False)
# #         for weight in self.decoder.weights:
# #             self.trainable_variables += [weight]

#         # Define classification layer (LOGIT OUTPUT)
#         self.classifier = tf.keras.layers.Dense(self.vocab_size)
# #         for weight in self.classifier.weights:
# #             self.trainable_variables += [weight]

#     def call(self, encoded_images, captions):
#         # TODO:
#         # 1) Embed the encoded images into a vector of the correct dimension for initial state
#         # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
#         # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
#         img_emb = self.image_embedding(encoded_images)
#         cap_emb = self.embedding(captions)
#         output = self.decoder(cap_emb, initial_state=img_emb)
#         logits = self.classifier(output)
#         return logits


# ########################################################################################

# class TransformerDecoder(tf.keras.Model):

#     def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

#         super().__init__(**kwargs)
#         self.vocab_size  = vocab_size
#         self.hidden_size = hidden_size
#         self.window_size = window_size

#         # TODO: Define image and positional encoding, transformer decoder, and classification layers

#         # Define feed forward layer to embed image features into a vector 
#         self.image_embedding = tf.keras.layers.Dense(self.hidden_size)

#         # Define positional encoding to embed and offset layer for language:
#         self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)

#         # Define transformer decoder layer:
#         self.decoder = TransformerBlock(self.hidden_size, True)
        
# #         self.encoder = EncoderLayer(d_model=128, num_heads=8, dff=64)

#         # Define classification layer (logits)
#         self.classifier = tf.keras.layers.Dense(self.vocab_size)
# #                                                 kernel_regularizer=tf.keras.regularizers.L1(0.0001),
# #                                                 activity_regularizer=tf.keras.regularizers.L2(0.0001))

#     def call(self, encoded_images, captions):
#         # TODO:
#         # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
#         # 2) Pass the captions through your positional encoding layer
#         # 3) Pass the english embeddings and the image sequences to the decoder
#         # 4) Apply dense layer(s) to the decoder out to generate logits
#         vec_encoded_images = self.image_embedding(encoded_images)
#         pos_captions = self.encoding(captions)
        
# #         temp = self.encoder(tf.concat([pos_captions, vec_encoded_images], axis=1))
# #         output = self.decoder(pos_captions, vec_encoded_images)
#         output = self.decoder(tf.concat([pos_captions, vec_encoded_images], axis=1), vec_encoded_images)

#         probs = self.classifier(output)
#         return probs[:,:pos_captions.shape[1],:]


import tensorflow as tf

try:
    from transformer import TransformerBlock, PositionalEncoding
except Exception as e:
    print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")


########################################################################################

class RNNDecoder(tf.keras.layers.Layer):
    
    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers
        
        # Define feed forward layer to embed image features into a vector with the models hidden size
        self.image_embedding = tf.keras.layers.Dense(hidden_size)
        
        # tf.keras.layers.GRU OR tf.keras.layers.LSTM for your RNN layer
        # tf.keras.layers.Embedding for the word embeddings
        # tf.keras.layers.Dense for all feed forward layers.
        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        
        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.GRU(hidden_size, return_sequences=True)
        
        # Define classification layer (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(vocab_size)
    
    def call(self, encoded_images, captions):
        # TODO:
        '''
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        '''
        image_embeddings = self.image_embedding(encoded_images)
        caption_embeddings = self.embedding(captions)
        decoder_output = self.decoder(caption_embeddings, initial_state=image_embeddings)
        logits = self.classifier(decoder_output)
        return logits


########################################################################################

class TransformerDecoder(tf.keras.Model):
    
    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        # TODO: Define image and positional encoding, transformer decoder, and classification layers
        
        # Define feed forward layer to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(hidden_size)
        
        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)
        
        # Define transformer decoder layer:
        self.decoder = TransformerBlock(hidden_size)
        
        # Define classification layer (logits)
        self.classifier = tf.keras.layers.Dense(vocab_size)
    
    def call(self, encoded_images, captions):
        # print('encoded_images', encoded_images.shape)
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        image_embeddings = self.image_embedding(encoded_images)
        caption_embeddings = self.encoding(captions)
        # print('image_embeddings', image_embeddings.shape)
        # print('caption_embeddings', caption_embeddings.shape)
        decoder_output = self.decoder(caption_embeddings, image_embeddings)
        logits = self.classifier(decoder_output)
        return logits