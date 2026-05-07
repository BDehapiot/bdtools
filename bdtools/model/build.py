#%% Imports -------------------------------------------------------------------

import segmentation_models as sm

# bdtools
from bdtools.model import metrics

# tensorflow
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

#%% Class(Build) --------------------------------------------------------------

class Build:
    
    def __init__(self, main, model_type="sm"):
        self.main = main
        self.model_type = model_type
        self.parameters = main.parameters
        for key, val in self.parameters.items():
            setattr(self, key, val)
            
        # Run
        if self.model_type == "sm":
            self.build_sm()
        if self.model_type == "aec":
            self.build_aec()
            
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=getattr(metrics, self.loss),
            metrics=[getattr(metrics, self.metric)],
            )
        
        # Pass attributes to main class
        self.main.model = self.model
        if self.model_type == "aec":
            self.main.model_enc = self.model_enc
            self.main.model_dec = self.model_dec
    
#%% Class(Build) build_sm() ---------------------------------------------------

    def build_sm(self):
        
        self.model = sm.Unet(
            self.backbone, 
            input_shape=self.input_shape,
            classes=1, # Parameter
            activation=self.activation,
            encoder_weights=None,
            )
            
#%% Class(Build) build_aec() --------------------------------------------------

    def build_aec(self):

        # Encoder -------------------------------------------------------------
        
        enc_inputs = layers.Input(shape=self.input_shape)
        x = enc_inputs
    
        for f in self.filters:
            x = layers.Conv2D(f, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            # x = layers.LeakyReLU(alpha=0.1)(x)
            x = layers.MaxPooling2D((2, 2))(x)
    
        x = layers.Flatten()(x)
        latent_space = layers.Dense(
            self.latent_size, activation="relu", name="latent_features")(x)

        self.model_enc = Model(enc_inputs, latent_space, name="encoder")
    
        # Decoder -------------------------------------------------------------
        
        dec_inputs = layers.Input(shape=(self.latent_size,))
        sdim = self.input_shape[0] // 2 ** len(self.filters)
        
        x = layers.Dense(sdim * sdim * self.filters[-1], activation="relu")(dec_inputs)
        x = layers.Reshape((sdim, sdim, self.filters[-1]))(x)
        for f in reversed(self.filters):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
    
        dec_outputs = layers.Conv2D(
            self.input_shape[-1], (3, 3), activation=self.activation, padding="same")(x)
        
        self.model_dec = Model(dec_inputs, dec_outputs, name="decoder")
    
        # Autoencoder ---------------------------------------------------------
        
        aec_inputs = layers.Input(shape=self.input_shape)
        enc_img = self.model_enc(aec_inputs)
        dec_img = self.model_dec(enc_img)
        
        self.model = Model(aec_inputs, dec_img, name="autoencoder")
    