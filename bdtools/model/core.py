#%% Imports -------------------------------------------------------------------

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
import numpy as np
from pathlib import Path
import segmentation_models as sm

# bdtools 
import metrics
from bdtools.norm import norm_pct
from bdtools.model.callbacks import CallBacks

# Tensorflow
from tensorflow.keras.optimizers import Adam

#%% Function(s) ---------------------------------------------------------------

def split(X, y, split=0.2):
    n_total = X.shape[0]
    n_val = int(n_total * split)
    idx = np.random.permutation(np.arange(0, n_total))
    X_trn = X[idx[n_val:]] 
    y_trn = y[idx[n_val:]]
    X_val = X[idx[:n_val]]
    y_val = y[idx[:n_val]]
    return X_trn, y_trn, X_val, y_val

#%% Class(UNet) ---------------------------------------------------------------

class UNet:
    
    def __init__(self, X, y, parameters, X_val=None, y_val=None):
        
        # Fetch
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Execute
        self.initialize()
        self.build()
        
#%% Class(UNet) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Paths
        if self.root_path is None:
            self.root_path = Path.cwd()
        self.model_name = (
            f"model_"
            f"{'x'.join(str(s) for s in self.X.shape)}"
            )
        self.model_path = Path(self.root_path / self.model_name)
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)
                        
#%% Class(UNet) : build() -----------------------------------------------------

    def build(self):
        
        self.model = sm.Unet(
            self.backbone, 
            input_shape=(None, None, 1), # Parameter
            classes=1,
            activation=self.activation,
            encoder_weights=None,
            )
    
#%% Class(UNet) : train() -----------------------------------------------------
    
    def train(self):
        
        # Split data
        if self.X_val is None:
            self.X_trn, self.y_trn, self.X_val, self.y_val = split(
                self.X, self.y, split=self.validation_split)
        else:
            self.X_trn, self.y_trn = self.X, self.y
            
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy", # Parameter
            metrics=[getattr(metrics, self.metric)],
            )

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
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import napari
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
    dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    data_path = Path.cwd().parent.parent / "_local" / dataset
    raw_trn_paths = list(data_path.rglob("*raw_trn.tif"))
    msk_trn_paths = list(data_path.rglob("*msk_trn.tif"))
    raw_val_paths = list(data_path.rglob("*raw_val.tif"))
    msk_val_paths = list(data_path.rglob("*msk_val.tif"))
    
    # Load data
    raw_trn = load_data(raw_trn_paths)
    msk_trn = load_data(msk_trn_paths)
    raw_val = load_data(raw_val_paths)
    msk_val = load_data(msk_val_paths)
    if "nuclei_semantic" in dataset:
        msk_trn = prep_mask(msk_trn)
        msk_val = prep_mask(msk_val)
        
    # Preprocess --------------------------------------------------------------
    
    raw_trn = norm_pct(raw_trn)
    msk_trn = (msk_trn > 0).astype("float32")
    
    # Display -----------------------------------------------------------------
    
    # # Display
    # idx = 0
    # vwr = napari.Viewer()
    # if isinstance(raw_trn, list):
    #     vwr.add_image(raw_trn[idx], visible=1, opacity=0.5)
    #     vwr.add_image(msk_trn[idx], visible=1, opacity=0.5)
    # else:
    #     vwr.add_image(raw_trn, visible=1, opacity=0.5)
    #     vwr.add_image(msk_trn, visible=1, opacity=0.5)
    
#%% Unet() --------------------------------------------------------------------

    parameters = {
        
        # Build
        "root_path"        : None,
        "model_name"       : None,
        "backbone"         : "resnet18",
        "activation"       : "sigmoid",
            
        # Train
        "epochs"           : 300,
        "batch_size"       : 4,
        "validation_split" : 0.2,
        "metric"           : "soft_dice_coef",
        "learning_rate"    : 0.001,
        "patience"         : 100,

        }

    unet = UNet(raw_trn, msk_trn, parameters)
    unet.train()
    # X_trn = unet.X_trn
    # y_trn = unet.y_trn
    # X_val = unet.X_val
    # y_val = unet.y_val
    
    
    