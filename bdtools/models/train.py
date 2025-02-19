#%% Imports -------------------------------------------------------------------

import pickle
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import segmentation_models as sm

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Skimage
from skimage.transform import downscale_local_mean, rescale

#%%




#%%

class UNet:
       
    def __init__(
            self, X, y,
            save_name="",
            save_path=Path.cwd(),
            downscale_steps=0, 
            backbone="resnet18",
            epochs=100,
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
        
        # Model name
        self.date = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        if not self.save_name:
            self.save_name = f"model_{self.date}"
        else:
            self.save_name = f"model_{self.save_name}"
        
        # Saving directory
        self.save_path = Path(Path.cwd(), self.save_name)
        self.backup_path = Path(Path.cwd(), f"{self.save_name}_backup")
        if self.save_path.exists():
            if self.weights_path and self.weights_path.exists():
                if self.backup_path.exists():
                    shutil.rmtree(self.backup_path)
                shutil.copytree(self.save_path, self.backup_path)
            shutil.rmtree(self.save_path)
        self.save_path.mkdir(exist_ok=True)
        
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
        
        self.custom_callbacks = UNet.CustomCallbacks(self)
        
        self.callbacks = [
            self.checkpoint, 
            self.early_stopping, 
            self.custom_callbacks,
            ]
        
    def train(self):
    
        self.history = self.model.fit(
            x=self.X_trn, y=self.y_trn,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=0,
            ) 
        
        plot_losses(
            self.custom_callbacks.trn_losses,
            self.custom_callbacks.val_losses,
            )
        
    class CustomCallbacks(Callback):
        
        def __init__(self, unet):
            super(UNet.CustomCallbacks, self).__init__()
            self.unet = unet
            self.trn_losses  = []
            self.val_losses  = []
            self.epoch_times = []
    
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
    
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Fetch 
            epochs = self.unet.epochs
            trn_loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            self.trn_losses.append(trn_loss)
            self.val_losses.append(val_loss)
            wait = self.unet.early_stopping.wait
            patience = self.unet.early_stopping.patience
            
            # Initialize
            epochs_nd = len(str(epochs))
            patience_nd = len(str(epochs))
            self.min_val_loss = np.min(self.val_losses)
            
            # Print
            print(
                f"Epoch {epoch:>{epochs_nd}}/{epochs} | "
                f"loss: {trn_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"wait: {wait:>{patience_nd}}/{patience} "
                f"({self.min_val_loss:.4f})"
                )
            
        def on_train_end(self, logs=None):
            trainable_params = sum(
                tf.keras.backend.count_params(w) 
                for w in self.model.trainable_weights
                )

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
        save_name="test",
        downscale_steps=3,
        )
    X = unet.X
    y = unet.y
    unet.train()

    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(X)
    # viewer.add_image(y)   

#%%

    # def plot_losses(unet):
        
    #     # Fetch
    #     epochs = unet.epochs
    #     trn_losses = unet.custom_callbacks.trn_losses
    #     val_losses = unet.custom_callbacks.val_losses
    #     nparams = unet.history["trainable_params"][0]

    #     # Initialize
    #     epoch_time = np.cumsum(np.array(unet.history["epoch_time"]))
    #     train_time = epoch_time[-1]
    #     bvl = np.min(unet.history["val_loss"])
    #     bvl_idx = np.argmin(unet.history["val_loss"])
    #     bvl_time = epoch_time[bvl_idx]  
        
    #     # Plot
    #     fig, axis = plt.subplots(1, 1, figsize=(6, 6))   
    #     axis.plot(trn_losses, label="loss")
    #     axis.plot(val_losses, label="val_loss")
    #     axis.axvline(x=bvl_idx, color="k", linestyle=":", linewidth=1)
    #     axis.axhline(y=bvl, color="k", linestyle=":", linewidth=1)
        
    #     axis.text(
    #         bvl_idx / epochs, 1.05, f"{bvl_time:.2f}s", size=10, color="k",
    #         transform=axis.transAxes, ha="center", va="center",
    #         )

    # plot_losses(unet)