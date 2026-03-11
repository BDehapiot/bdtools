#%% Imports -------------------------------------------------------------------

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

from pathlib import Path
import segmentation_models as sm

# bdtools
from bdtools.check import Check_parameter
from bdtools.unet import metrics
from bdtools.unet.prepare import Prepare
from bdtools.unet.callbacks import CallBacks

# tensorflow
from tensorflow.keras.optimizers import Adam

#%% Class(UNet) ---------------------------------------------------------------

class UNet:
    
    def __init__(self, X, y, parameters):
        self.X, self.y = X, y
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self.initialize()
        Prepare(self)
        
#%% Class(UNet) initialize() --------------------------------------------------

    def initialize(self):
        
        # Check data
        Check_parameter(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )
        if self.X[0].ndim == 2:
            self.input_shape = (None, None, 1)
        elif self.X[0].ndim == 3:
            self.input_shape = (None, None, 1)
        
        # Model name
        if self.root_path is None:
            self.root_path = Path.cwd()
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
                f"{self.augment_iterations}-{n_trn}"
                )
        self.model_path = self.root_path / self.model_name
        self.weights_path = self.model_path / "weights.h5"
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)
        
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
        if self.weights_path.exists():
            print("load model weights")
            self.model.load_weights(self.weights_path)
        
#%% Class(UNet) train() -------------------------------------------------------

    def train(self):

        self.build()    
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

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import numpy as np
    from skimage import io
    from pathlib import Path
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
        "model_name"         : "model_128_binary_500-80",

        # Build
        "backbone"           : "resnet18",
        "activation"         : "sigmoid",
            
        # Train
        "epochs"             : 128,
        "batch_size"         : 4,
        "validation_split"   : 0.2,
        "metric"             : "soft_dice_coef",
        "learning_rate"      : 0.001,
        "patience"           : 64,

        # Prepare
        "multichannel"       : True,
        "patch_size"         : 128,
        "patch_overlap"      : 64,
        "mask_method"        : "binary",
                
        # Augment
        "augment_iterations" : 500,
        "augment_invert_p"   : 0,
        "augment_gamma_p"    : 0.5,
        "augment_gblur_p"    : 0.5,
        "augment_noise_p"    : 0.5,
        "augment_flip_p"     : 0.5,
        "augment_distord_p"  : 0.5,

        }
    
    unet = UNet(raw_trn, msk_trn, parameters)
    # unet.train()
    X_patches = unet.X_patches
    y_patches = unet.y_patches
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(np.stack(X_patches))
    vwr.add_image(np.stack(y_patches))