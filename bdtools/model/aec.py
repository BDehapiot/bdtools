#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from pathlib import Path

# bdtools
from bdtools.check import Check
from bdtools.model.prepare import Prepare
from bdtools.model import metrics
from bdtools.model.callbacks import CallBacks

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

#%% Class(AEC) ----------------------------------------------------------------

class AutoEncoder:
    
    def __init__(self, parameters=None, model_path=None):
        self.parameters = parameters
        self.model_path = model_path
        
        # Run
        self.initialize()

#%% Class(AEC) Function(s) ----------------------------------------------------

    def get_parameters(self):
        for key, val in self.parameters.items():
            setattr(self, key, val)
        
    def get_model_path(self):
        if self.root_path is None:
            self.root_path = Path.cwd()
        self.model_path = self.root_path / self.model_name
        
    def load_model_weights(self):
        self.weights_path = self.model_path / "weights.keras"
        if self.weights_path.exists():
            print(f"({self.model_path.name}) : load weights ")
            self.model.load_weights(self.weights_path)

#%% Class(AEC) initialize() ---------------------------------------------------

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

        self.get_parameters()
                    
    def initialize_train(self):
                            
        if self.model_path is None:

            # Default model name (if not provided)
            if self.model_name is None:
                
                n = self.X.shape[0]
                n_trn = int(n - (n * self.validation_split))
                self.model_name = (
                    "model-aec_"
                    f"{self.input_shape[0]}_"
                    f"{len(self.filters)}_"
                    f"{self.latent_size}_"
                    f"{n_trn}-"
                    f"{self.augment_iterations}"
                    )
                
                self.get_model_path()
                
            else:
                
                self.get_model_path()
                self.load_model_weights()
                                
        else:
            
            self.load_model_weights()
        
        # Create model directory
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)

        # Save parameters
        with open(self.model_path / "parameters.pkl", "wb") as file:
            pickle.dump(self.parameters, file)

#%% Class(AEC) build() --------------------------------------------------------

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
        
        self.model_aec = Model(aec_inputs, dec_img, name="autoencoder")
    
        # Compile -------------------------------------------------------------
        
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            dice_loss = 1 - metrics.soft_dice_coef(y_true, y_pred)
            return bce + (5.0 * dice_loss)
        
        self.model_aec.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=combined_loss,
            metrics=[getattr(metrics, self.metric)],
            )
        
#%% Class(AEC) prepare() ------------------------------------------------------

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
                
        # Check
        self.X = X
        Check(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )
        
        # Prepare
        Prepare(self, self.X, y=None)
        
        # Format datasets (specific to aec)
        self.X_trn_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_trn, self.X_trn))
        self.X_trn_dataset = self.X_trn_dataset.batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.X_val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_val, self.X_val))
        self.X_val_dataset = self.X_val_dataset.batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Display (optional)
        if self.display:
            import napari
            vwr = napari.Viewer()
            vwr.add_image(self.X_trn, name="X_trn")
            vwr.grid.enabled = True
            return
        
        # Build
        self.build()
        
        # Initialize training
        self.initialize_train()

        # Callbacks
        self.callbacks = [CallBacks(self)]
                
        try:
                    
            # Train    
            self.history = self.model_aec.fit(
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
            self.model_aec.stop_training = True
            for cb in self.callbacks:
                cb.on_train_end(logs={})
        
#%% Class (AEC) predict() -----------------------------------------------------

    def predict(self, X, patch_overlap=None, batch_size=32, chunk_size=None):
    
        self.build()
        
        # Initialize
        if patch_overlap is None:
            patch_overlap = self.patch_size // 2
        multichannel = True if self.input_shape[-1] > 1 else False
        Check(X, name="X", ctype=(np.ndarray, list), dtype=float, vrange=(0, 1))
        
        # Convert X to list
        if isinstance(X, list):
            islist = True
        else:
            islist = False
            X = [X]
        
        # prds = []
        # for arr in X:
            
        #     shape = arr.shape[:-1] if multichannel else arr.shape
                    
        #     # Predict
            
        #     if chunk_size:
                
        #         n_chunks = int(np.ceil(arr.shape[0] / chunk_size))
        #         print(f"n_chunks = {n_chunks}")
                
        #         prd = []
        #         for i, idx in enumerate(range(0, len(arr), chunk_size)):
        #             chunk = arr[idx:idx + chunk_size]
        #             prd.append(
        #                 self.model_aec.predict(chunk, batch_size=batch_size))
        #         prd = np.concatenate(prd, axis=0)
                
        #         del chunk
            
        #     else:
                
        #         prd = self.model_aec.predict(arr, batch_size=batch_size)
                
        #         # del patches
                
        #     # Merge patches
        #     prd = prd.squeeze()
        #     prds.append(prd)
    
        # return prds if islist else prds[0]
 
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import numpy as np
    from skimage import io
    from skimage.measure import label
    
    # -------------------------------------------------------------------------
    
    def load_data(paths):
        data = []
        for path in paths:
            data.append(io.imread(path))
        if len(data) == 1:
            data = np.stack(data).squeeze()
        return data
    
    def prep_mask(msk):
        msk_1 = label(msk == 1)
        msk_2 = label(msk == 2)
        msk_3 = label(msk == 3)
        msk_2[msk_2 > 0] += np.max(msk_1)
        msk_3[msk_3 > 0] += np.max(msk_2)
        return msk_1 + msk_2 + msk_3
    
    # Load --------------------------------------------------------------------
    
    # Paths
    
    # dataset = "em_mito"
    # dataset = "fluo_tissue"
    dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    # dataset = "sat_roads"
    
    data_path = Path.cwd().parent.parent / "_local" / dataset
    raw_trn_paths = list(data_path.rglob("*raw_trn.tif"))
    msk_trn_paths = list(data_path.rglob("*msk_trn.tif"))
    
    # Load data
    raw_trn = load_data(raw_trn_paths)
    msk_trn = load_data(msk_trn_paths)
    if "nuclei_semantic" in dataset:
        msk_trn = prep_mask(msk_trn)
        
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # if isinstance(raw_trn, list):
    #     idx = 2
    #     vwr.add_image(raw_trn[idx])
    #     vwr.add_image(msk_trn[idx])
    # else:
    #     vwr.add_image(raw_trn)
    #     vwr.add_image(msk_trn)
        
    # Normalization -----------------------------------------------------------
    
    from bdtools.norm import norm_pct
        
    if "sat_roads" in dataset:
        raw_trn = raw_trn.astype("float32") / 255
    else:
        raw_trn = norm_pct(
            raw_trn, pct_low=0.01, pct_high=99.9, sample_fraction=1)
    
#%% AEC() train() -------------------------------------------------------------
    
    parameters = {

        # Paths
        "root_path"          : None,
        "model_name"         : None,

        # Build
        "input_shape"      : (128, 128, 1),
        "filters"          : [32, 64],
        "latent_size"      : 512,
        "activation"       : "sigmoid",
            
        # Train
        "display"            : 0,
        "epochs"             : 256,
        "batch_size"         : 16,
        "validation_split"   : 0.2,
        "metric"             : "soft_dice_coef",
        "learning_rate"      : 0.001,
        "patience"           : 64,

        # Prepare
        "patch_size"         : 128,
        "patch_overlap"      : 0,
                
        # Augment
        "augment_iterations" : None,
        "augment_gamma_p"    : 0,
        "augment_gblur_p"    : 0,
        "augment_noise_p"    : 0.5,
        "augment_flip_p"     : 0.5,
        "augment_distort_p"  : 0.5,
        "augment_params"     : {
            
            # Gamma
            "gamma_low"          : 0.75,
            "gamma_high"         : 1.25,
            "gamma_chn"          : "independent",
            
            # Gaussian blur
            "gblur_sigma_low"    : 1,
            "gblur_sigma_high"   : 3,
            "gblur_chn"          : "shared",
            
            # Noise
            "noise_gain_low"     : 30,
            "noise_gain_high"    : 60,
            "noise_std_low"      : 3,
            "noise_std_high"     : 6,
            "noise_chn"          : "independent",
            
            # Grid distort
            "distort_steps_low"  : 1,
            "distort_steps_high" : 10,
            "distort_limit_low"  : 0.1,
            "distort_limit_high" : 0.5,
            
            },

        }
    
    # aec = AutoEncoder(parameters=parameters, model_path=None)
    # aec.train(raw_trn)
        
#%% AEC() Predict() -----------------------------------------------------------
    
    model_path = Path(Path.cwd(), "model-aec_128_2_512_35")
    aec = AutoEncoder(parameters=None, model_path=model_path)
    aec.predict(raw_trn)
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # if isinstance(raw_trn, list):
    #     idx = 2
    #     vwr.add_image(raw_trn[idx])
    #     vwr.add_image(prds[idx])
    # else:
    #     vwr.add_image(raw_trn)
    #     vwr.add_image(prds)