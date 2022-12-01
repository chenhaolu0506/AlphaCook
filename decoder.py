import tensorflow as tf

from transformer import TransformerBlock, PositionalEncoding
# except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        # with the models hidden size
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size)

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size)

        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)

        # Define classification layer (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        enbedded_images = self.image_embedding(encoded_images)
        embedded_english = self.embedding(captions)
        decoder_outputs = self.decoder(embedded_english, initial_state=enbedded_images)
        logits = self.classifier(decoder_outputs)
        return logits

    def get_config(self):
        return {k:getattr(self, k) for k in "vocab_size hidden_size window_size".split()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        self.image_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_size, activation="relu"),
                tf.keras.layers.Dense(self.hidden_size),
            ]
        )

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(self.hidden_size)

        # Define classification layer (logits)
        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_size, activation="relu"),
                tf.keras.layers.Dense(self.vocab_size),
            ]
        )

    def call(self, encoded_images, captions):
        # encoded images = batch size X 1 X size
        # captions batch size X window size
        # batch size X window size X vocab size
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # add positional embeddings to the word embeddings
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        enbedded_images = self.image_embedding(tf.expand_dims(encoded_images, 1))
        x = self.encoding(captions)
        x = self.decoder(x, enbedded_images)
        probs = self.classifier(x)
        return probs

