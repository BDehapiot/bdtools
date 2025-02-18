#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
import segmentation_models as sm

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Skimage
from skimage.transform import downscale_local_mean, rescale

#%% 

class UNet:
       
    def __init__(
            self, X, y,
            save_name="",
            save_path=Path.cwd(),
            
            
            downscale_steps=0,
            
            backbone="resnet18",
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            learning_rate=0.001,
            patience=20,
            weights_path="",
            ):
        
        # Fetch
        self.X, self.y = X, y
        self.save_name = save_name
        self.save_path = save_path
        self.downscale_steps = downscale_steps
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        self.weights_path = weights_path
        
        # Initialize
        
        
        # Execute   
        self.downscale()
        self.split()
        self.build()        
        
    def downscale(self):
        if self.downscale_steps > 0:
            df = 2**self.downscale_steps
            self.X = downscale_local_mean(self.X, (1, df, df))
            self.y = downscale_local_mean(self.y, (1, df, df))
        
    def split(self):
        n_total = self.X.shape[0]
        n_val = int(n_total * self.validation_split)
        idx = np.random.permutation(np.arange(0, n_total))
        self.X_trn = self.X[idx[n_val:]] 
        self.y_trn = self.y[idx[n_val:]]
        self.X_val = self.X[idx[:n_val]]
        self.y_val = self.y[idx[:n_val]]
        
    def build(self):
        
        self.model = sm.Unet(
            self.backbone, 
            input_shape=(None, None, 1), 
            classes=1, 
            activation="sigmoid", 
            encoder_weights=None,
            )
        
        if self.weights_path:
            self.model.load_weights(
                Path(Path.cwd(), f"{self.save_name}_backup", "weights.h5"))
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy", 
            metrics=["mse"],
            )
        
        # Callbacks
       
        self.checkpoint = ModelCheckpoint(
            filepath=Path(self.save_path, "weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            )
        
        self.early_stopping = EarlyStopping(
            patience=self.patience, 
            monitor='val_loss',
            mode="min",
            )
        
        self.callbacks = CustomCallback(self)
        
    def train(self):
    
        self.history = self.model.fit(
            x=self.X_trn, y=self.y_trn,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=0,
            ) 
        
    class CustomCallback(Callback):
        
        def __init__(self, unet):
            
            super(UNet.CustomCallback, self).__init__()
            
            # Fetch
            self.unet = unet

            # Initialize
            self.trn_loss = []
            self.val_loss = []

            # Checkpoints
            self.checkpoint = ModelCheckpoint(
                filepath=Path(self.unet.save_path, "weights.h5"),
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                )
            
            # Early stopping
            self.earlystopping = EarlyStopping(
                patience=self.unet.patience,
                monitor="val_loss",
                mode="min",
                )
    
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            
            # Fetch 
            
            trn_loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            wait = self.unet.earlystopping.wait
            patience = self.unet.earlystopping.patience
            
            # Monitor losses
            trn_loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            self.trn_loss.append(trn_loss)
            self.val_loss.append(val_loss)
            
            # Monitor patience
            wait = self.unet.earlystopping.wait
            patience = self.unet.earlystopping.patience
            print(
                f"Epoch {epoch:03d}/{self.unet.epochs:03d} - "
                f"loss: {trn_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"patience: {wait}/{patience} ({np.min(self.val_loss):.4f})"
                )

#%% Class: CustomCallback -----------------------------------------------------

# class CustomCallback(Callback):
    
#     def __init__(self, unet):
        
#         super(CustomCallback, self).__init__()
#         self.unet = unet
#         self.trn_loss, self.val_loss = [], []
        
#     def on_epoch_end(self, epoch, logs=None):
        
#         # Fetch loss
#         trn_loss = logs["loss"]
#         val_loss = logs.get("val_loss")
#         self.trn_loss.append(trn_loss)
#         self.val_loss.append(val_loss)
        
#         # Print
#         print(
#             f"epoch {epoch:03d}, "
#             f"loss: {trn_loss:.4f}, "
#             f"val_loss: {val_loss:.4f}"
#             )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path
    from bdtools.models import preprocess

    # Parameters
    dataset = "em_mito"
    # dataset = "fluo_nuclei"
    patch_size = 256
    patch_overlap = 0
    
    # Paths
    local_path = Path.cwd().parent.parent / "_local"
    img_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
    msk_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    
    # Load images & masks
    X = io.imread(img_path)
    y = io.imread(msk_path)
    
    # Preprocess
    t0 = time.time()
    print("preprocess :", end=" ", flush=True)
    X, y = preprocess(
        X, msks=y, 
        img_norm="global",
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Train
    unet = UNet(
        X, y, 
        downscale_steps=2,
        )
    X = unet.X
    y = unet.y
    unet.train()
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(X)
    # viewer.add_image(y)   

    pass