#%% Imports -------------------------------------------------------------------

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

import pickle
from pathlib import Path
import segmentation_models as sm

# bdtools
from bdtools.check import Check
from bdtools.unet import metrics
from bdtools.unet.prepare import Prepare
from bdtools.unet.callbacks import CallBacks
from bdtools.patch import get_patches, merge_patches

# tensorflow
from tensorflow.keras.optimizers import Adam

#%% Class(UNet) ---------------------------------------------------------------

class UNet:
    
    def __init__(self, parameters=None, model_path=None):
        self.model_path = model_path
        self.parameters = parameters
        
        # Run
        self.initialize()
        self.build()

#%% Class(UNet) initialize() --------------------------------------------------

    def initialize(self):
        
        # Check inputs
        if (self.parameters is None) == (self.model_path is None):
            raise ValueError(
                "User must either provide 'parameters' or 'model_path', but not both"
                )
                    
        # Parameters
        if self.parameters is None:
            self.params_path = self.model_path / "parameters.pkl"
            with open(self.params_path, "rb") as file:
                self.parameters = pickle.load(file)
                print(f"({self.model_path.name}) : load parameters ")
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
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
        if self.model_path is not None:
            print(f"({self.model_path.name}) : load weights ")
            self.model.load_weights(self.weights_path)
                        
#%% Class(UNet) train() -------------------------------------------------------

    def train(self, X, y):
        
        # Initialize
        self.X, self.y = X, y
        Check(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )
        self.get_parameters()
        Prepare(self)
        
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

    def predict(self, X):
        
        # Initialize
        Check(
            X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )

        # Prepare patches
        multichannel = True if self.input_shape[-1] > 1 else False
        if isinstance(X, list):
            prds = []
            for arr_X in X:
                X_patches = get_patches(
                    arr_X, self.patch_size, self.patch_overlap, 
                    multichannel=multichannel
                    )
                X_patches = np.stack(X_patches)
                prd = self.model.predict(X_patches, verbose=1).squeeze()
                out_shape = arr_X.shape[:-1] if multichannel else arr_X.shape
                prd = merge_patches(prd, out_shape, self.patch_overlap)
                prds.append(prd)
        if isinstance(X, np.ndarray):
            X_patches = get_patches(
                X, self.patch_size, self.patch_overlap, 
                multichannel=multichannel
                )
            X_patches = np.stack(X_patches)
            prds = self.model.predict(X_patches, verbose=1).squeeze()
            out_shape = X.shape[:-1] if multichannel else X.shape
            prds = merge_patches(prds, out_shape, self.patch_overlap)
                
        return prds
        
#%% Class(UNet) funtion(s) ----------------------------------------------------

    def get_parameters(self):
        
        # Model name 
        if self.model_name is None:
            if isinstance(self.X, list):
                n = len(self.X)
            elif isinstance(self.X, np.ndarray):
                n = self.X.shape[0]
            n_trn = int(n - (n * self.validation_split))
            self.model_name = (
                "model_"
                f"{self.patch_size}_"
                f"{self.mask_method}_"
                f"{self.augment_iterations}"
                f"-{n_trn}"
                )
            
        # Paths
        if self.root_path is None:
            self.root_path = Path.cwd()
        self.model_path = self.root_path / self.model_name
        self.params_path = self.model_path / "parameters.pkl"
        self.weights_path = self.model_path / "weights.h5"
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)
            
        # Append parameters
        self.parameters["model_name"  ] = self.model_name
        self.parameters["root_path"   ] = self.root_path
        self.parameters["model_path"  ] = self.model_path
        self.parameters["params_path" ] = self.params_path
        self.parameters["weights_path"] = self.weights_path
        self.parameters["input_shape" ] = self.input_shape
        
        # Save parameters
        with open(self.params_path, "wb") as file:
            pickle.dump(self.parameters, file)
            
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
    # dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    dataset = "sat_roads"
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
        
    test = raw_trn[2]
    
        
    # Normalization -----------------------------------------------------------
    
    # from bdtools.norm import norm_pct
        
    # if "sat_roads" in dataset:
    #     raw_trn = raw_trn.astype("float32") / 255
    # else:
    #     raw_trn = norm_pct(
    #         raw_trn, pct_low=0.01, pct_high=99.9, sample_fraction=1)
    
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
        "epochs"             : 256,
        "batch_size"         : 16,
        "validation_split"   : 0.2,
        "metric"             : "soft_dice_coef",
        "learning_rate"      : 0.001,
        "patience"           : 64,

        # Prepare
        "patch_size"         : 128,
        "patch_overlap"      : 0,
        "mask_method"        : "binary",
                
        # Augment
        "augment_iterations" : 0,
        "augment_invert_p"   : 0,
        "augment_gamma_p"    : 0.5,
        "augment_gblur_p"    : 0.5,
        "augment_noise_p"    : 0.5,
        "augment_flip_p"     : 0.5,
        "augment_distord_p"  : 0.5,

        }
    
    # Train() -----------------------------------------------------------------
    
    # unet = UNet(parameters=parameters, model_path=None)
    # unet.train(raw_trn, msk_trn)
    
    # Predict() ---------------------------------------------------------------
    
    # model_path = Path(Path.cwd(), "model_128_binary_0-35")
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