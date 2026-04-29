#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from pathlib import Path

# bdtools
from bdtools.check import Check
from bdtools.autoencoder import metrics
from bdtools.autoencoder.callbacks import CallBacks

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

#%% Class(AutoEncoder) --------------------------------------------------------

class AutoEncoder:
    
    def __init__(self, parameters=None, model_path=None):
        self.parameters = parameters
        self.model_path = model_path
        
        # Run
        self.initialize()

#%% Class(AutoEncoder) initialize() -------------------------------------------

    def initialize(self):
        
        # Check inputs
        if (self.parameters is None) == (self.model_path is None):
            raise ValueError(
                "User must either provide 'parameters' or 'model_path', but not both"
                )
                    
        # Load parameters (if not provided)
        if self.parameters is None:
            
            with open(self.model_path / "parameters.pkl", "rb") as file:
                self.parameters = pickle.load(file)
                print(f"({self.model_path.name}) : load parameters ")

        for key, val in self.parameters.items():
            # if not isinstance(val, dict):
            setattr(self, key, val)
        
    def initialize_train(self):
        
        # Get model path (if not provided)
        if self.model_path is None:
            
            for key, val in self.parameters.items():
                # if not isinstance(val, dict):
                setattr(self, key, val)
            
            # Default model name (if not provided)
            if self.parameters["model_name"] is None:
                n = self.X.shape[0]
                n_trn = int(n - (n * self.validation_split))
                self.model_name = (
                    "model_"
                    f"{self.input_shape[0]}_"
                    f"{len(self.filters)}_"
                    f"{self.latent_size}_"
                    f"{n_trn}"
                    )
                
            if self.root_path is None:
                self.root_path = Path.cwd()
            self.model_path = self.root_path / self.model_name
        
        # Create model directory
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)
        
        # Save parameters
        with open(self.model_path / "parameters.pkl", "wb") as file:
            pickle.dump(self.parameters, file)

#%% Class(AutoEncoder) build() ------------------------------------------------

    def build(self):

        # Encoder -------------------------------------------------------------
        
        enc_inputs = layers.Input(shape=self.input_shape)
        x = enc_inputs
    
        for f in self.filters:
            x = layers.Conv2D(f, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D((2, 2))(x)
    
        x = layers.Flatten()(x)
        latent_space = layers.Dense(
            self.latent_size, activation="relu", name="latent_features")(x)
        
        self.encoder = Model(enc_inputs, latent_space, name="encoder")
    
        # Decoder -------------------------------------------------------------
        
        dec_inputs = layers.Input(shape=(self.latent_size,))
        sdim = self.crop_size // 2 ** len(self.filters)
        
        x = layers.Dense(sdim * sdim * self.filters[-1], activation="relu")(dec_inputs)
        x = layers.Reshape((sdim, sdim, self.filters[-1]))(x)
        for f in reversed(self.filters):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
    
        dec_outputs = layers.Conv2D(
            self.input_shape[-1], (3, 3), activation=self.activation, padding="same")(x)
        
        self.decoder = Model(dec_inputs, dec_outputs, name="decoder")
    
        # Autoencoder ---------------------------------------------------------
        
        aec_inputs = layers.Input(shape=self.input_shape)
        enc_img = self.encoder(aec_inputs)
        dec_img = self.decoder(enc_img)
        
        self.autoencoder = Model(aec_inputs, dec_img, name="autoencoder")
    
        # Compile -------------------------------------------------------------
        
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            dice_loss = 1 - metrics.soft_dice_coef(y_true, y_pred)
            return bce + (5.0 * dice_loss)
        
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=combined_loss,
            metrics=[getattr(metrics, self.metric)],
            )
        
#%% Class(AutoEncoder) prepare() ----------------------------------------------

    def prepare(self):
        
        def split_data(X):
            n_total = X.shape[0]
            n_val = int(n_total * self.validation_split)
            idx = np.random.permutation(np.arange(0, n_total))
            X_trn = X[idx[n_val:]] 
            X_val = X[idx[:n_val]]
            return X_trn, X_val
        
        # Split and setup data pipeline         
        self.X_trn, self.X_val = split_data(self.X)
        self.X_trn_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_trn, self.X_trn))
        self.X_trn_dataset = self.X_trn_dataset.batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.X_val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_val, self.X_val))
        self.X_val_dataset = self.X_val_dataset.batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)
        
#%% Class(AutoEncoder) train() ------------------------------------------------ 
        
    def train(self, X):
                
        # Initialize
        self.X = X
        Check(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )
        self.initialize_train()
        self.build()
        self.prepare()
        
        # Callbacks
        self.callbacks = [CallBacks(self)]
                
        try:
                    
            # Train    
            self.history = self.autoencoder.fit(
                x=self.X_trn_dataset,
                validation_data=self.X_val_dataset,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.callbacks,
                verbose=0,
                )
        
        # Interrupt
        except KeyboardInterrupt:
            print("Training interrupted.")
            self.autoencoder.stop_training = True
            for cb in self.callbacks:
                cb.on_train_end(logs={})
                
#%% Class (AutoEncoder) predict() ---------------------------------------------

    def predict(self, X, batch_size=32, chunk_size=None):
    
        self.build()
        
        # Initialize
        multichannel = True if self.input_shape[-1] > 1 else False
        Check(X, name="X", ctype=(np.ndarray, list), dtype=float, vrange=(0, 1))
        
        # Convert X to list
        if isinstance(X, list):
            islist = True
        else:
            islist = False
            X = [X]
            
        self.X = X # Debug
        
        prds = []
        for arr in X:
            
            shape = arr.shape[:-1] if multichannel else arr.shape
                    
            # Predict
            
            if chunk_size:
                
                n_chunks = int(np.ceil(arr.shape[0] / chunk_size))
                print(f"n_chunks = {n_chunks}")
                
                prd = []
                for i, idx in enumerate(range(0, len(arr), chunk_size)):
                    chunk = arr[idx:idx + chunk_size]
                    prd.append(
                        self.autoencoder.predict(chunk, batch_size=batch_size))
                prd = np.concatenate(prd, axis=0)
                
                del chunk
            
            else:
                
                prd = self.autoencoder.predict(arr, batch_size=batch_size)
                
                # del patches
                
            # Merge patches
            prd = prd.squeeze()
            prds.append(prd)
    
        return prds if islist else prds[0]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
     autoencoder_parameters = {
        
        # Paths
        "root_path"        : None,
        "model_name"       : None,
            
        # Build
        "input_shape"      : (32, 32, 7),
        "filters"          : [32, 64],
        "latent_size"      : 512,
        "activation"       : "sigmoid",
        
        # Train
        "epochs"           : 256,
        "batch_size"       : 256,
        "validation_split" : 0.2,
        "metric"           : "soft_dice_coef",
        "learning_rate"    : 0.001,
        "patience"         : 64,
                    
        }