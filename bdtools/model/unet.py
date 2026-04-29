#%% Imports -------------------------------------------------------------------

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

import pickle
import numpy as np
from pathlib import Path
import segmentation_models as sm

# bdtools
from bdtools.check import Check
from bdtools.model import metrics
from bdtools.model.prepare import Prepare
from bdtools.model.callbacks import CallBacks
from bdtools.patch import extract_patches, merge_patches

# tensorflow
from tensorflow.keras.optimizers import Adam

#%% Class(UNet) ---------------------------------------------------------------

class UNet:
    
    def __init__(self, parameters=None, model_path=None):
        self.parameters = parameters
        self.model_path = model_path
        
        # Run
        self.initialize()

#%% Class(UNet) initialize() --------------------------------------------------

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
                    f"{self.patch_size}_"
                    f"{self.mask_method}_"
                    f"{n_trn}-"
                    f"{self.augment_iterations}"
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
        
#%% Class(UNet) build() -------------------------------------------------------

    def build(self):
        
        # Build
        self.model = sm.Unet(
            self.backbone, 
            input_shape=self.input_shape,
            classes=1, # Parameter
            activation=self.activation,
            encoder_weights=None,
            )
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy", # Parameter
            metrics=[getattr(metrics, self.metric)],
            )
        
        # Load weights (optional)
        weights_path = self.model_path / "weights.keras"
        if weights_path.exists():
            print(f"({self.model_path.name}) : load weights ")
            self.model.load_weights(weights_path)
                        
#%% Class(UNet) train() -------------------------------------------------------

    def train(self, X, y):
        
        # Check
        self.X, self.y = X, y
        Check(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )
        
        # Prepare
        Prepare(self, self.X, y=self.y)
        
        # Display (optional)
        if self.display:
            import napari
            vwr = napari.Viewer()
            vwr.add_image(self.y_trn, name="y_trn")
            vwr.add_image(self.X_trn, name="X_trn")
            vwr.grid.enabled = True
            return
        
        # Initialize & build
        self.initialize_train()
        self.build()

        # Callbacks
        self.callbacks = [CallBacks(self)]
        
        try:
        
            # Train
            self.history = self.model.fit(
                x=self.X_trn, y=self.y_trn,
                validation_data=(self.X_val, self.y_val),
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=self.callbacks,
                verbose=0,
                )
        
        # Interrupt
        except KeyboardInterrupt:
            print("Training interrupted.")
            self.model.stop_training = True
            for cb in self.callbacks:
                cb.on_train_end(logs={})

#%% Class(Unet) predict() -----------------------------------------------------

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
                
        prds = []
        for arr in X:
            
            shape = arr.shape[:-1] if multichannel else arr.shape
            
            # Extract patches
            patches = np.stack(extract_patches(
                arr, self.patch_size, patch_overlap, 
                multichannel=multichannel
                ))
        
            # Predict
            
            if chunk_size:
                
                n_chunks = int(np.ceil(patches.shape[0] / chunk_size))
                print(f"n_chunks = {n_chunks}")
                
                prd = []
                for i, idx in enumerate(range(0, len(patches), chunk_size)):
                    chunk = patches[idx:idx + chunk_size]
                    prd.append(
                        self.model.predict(chunk, batch_size=batch_size))
                prd = np.concatenate(prd, axis=0)
                
                del chunk
            
            else:
                
                prd = self.model.predict(patches, batch_size=batch_size)
                
                del patches
                
            # Merge patches
            prd = prd.squeeze()
            prd = merge_patches(prd, shape, patch_overlap)
            prds.append(prd)
    
        return prds if islist else prds[0]
                    
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
    dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
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
    
#%% UNet() --------------------------------------------------------------------
    
    parameters = {

        # Paths
        "root_path"          : None,
        "model_name"         : None,

        # Build
        "input_shape"        : (None, None, 1),
        "backbone"           : "resnet18",
        "activation"         : "sigmoid",
            
        # Train
        "display"            : 0,
        "epochs"             : 256,
        "batch_size"         : 16,
        "validation_split"   : 0.2,
        "metric"             : "soft_dice_coef",
        "learning_rate"      : 0.001,
        "patience"           : 64,

        # Prepare
        "patch_size"         : 512,
        "patch_overlap"      : 256,
        "mask_method"        : "binary",
                
        # Augment
        "augment_iterations" : 128,
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
    
    # Train() -----------------------------------------------------------------
    
    unet = UNet(parameters=parameters, model_path=None)
    unet.train(raw_trn, msk_trn)
        
    # Predict() ---------------------------------------------------------------
    
    # model_path = Path(Path.cwd(), "model_256_binary_1280-2048")
    # unet = UNet(parameters=None, model_path=model_path)
    # prds = unet.predict(raw_trn)
    
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