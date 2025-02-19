#%% Imports -------------------------------------------------------------------

import pickle
import shutil
import numpy as np
from skimage import io
from pathlib import Path
from datetime import datetime
import segmentation_models as sm

# bdtools
import metrics

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Skimage
from skimage.transform import downscale_local_mean, rescale

# Matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

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
            metric="soft_dice_coef",
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
        self.metric = metric
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
            input_shape=(None, None, 1), # Parameter
            classes=1, # Parameter
            activation="sigmoid", # Parameter
            encoder_weights=None,
            )
        
        if self.weights_path:
            self.model.load_weights(
                Path(Path.cwd(), f"{self.save_name}_backup", "weights.h5"))
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy", # Parameter
            metrics=[getattr(metrics, self.metric)],
            )

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

        self.callbacks = [
            self.checkpoint, 
            self.early_stopping, 
            self.CustomCallbacks(self),
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
        
    def predict(self, X):
        return self.model.predict(X.squeeze())
        
    class CustomCallbacks(Callback):
        
        def __init__(self, unet):
            super(UNet.CustomCallbacks, self).__init__()
            
            # Fetch
            self.unet = unet    
            self.epochs = unet.epochs
            
            # Initialize
            self.trn_losses  = []
            self.val_losses  = []
            self.trn_metrics = []
            self.val_metrics = []
            self.epoch_times = []
            self.epoch_durations = []
            
        def print_log(self):
            
            # Fetch
            epoch = self.epoch
            epochs = self.epochs
            trn_loss = self.trn_losses[-1]
            val_loss = self.val_losses[-1]
            best_val_loss = self.best_val_loss
            trn_metric = self.trn_metrics[-1]
            val_metric = self.val_metrics[-1]
            wait = self.unet.early_stopping.wait
            patience = self.unet.patience

            # Print
            print(
                f"epoch {epoch:>{len(str(epochs))}}/{epochs} "
                f"wait {wait:>{len(str(patience))}}/{patience} "
                f"({best_val_loss:.4f}) "
                f"l|{trn_loss:.4f}| "
                f"vl|{val_loss:.4f}| "
                f"m|{trn_metric:.4f}| "
                f"vm|{val_metric:.4f}| "
                )
            
        def predict(self):
            
            # Predict
            prds = self.unet.predict(unet.X_val)
            
            # Plot
            plt.ioff() # turn off inline plot
            idxs = np.random.randint(0, prds.shape[0], size=20) 
            for i, idx in enumerate(idxs):
                            
                fig, (ax0, ax1, ax2) = plt.subplots(
                    nrows=1, ncols=3, figsize=(15, 5))
                cmap0, cmap1, cmap2 = cm.gray, cm.plasma, cm.plasma
                shrink = 0.75
                
                ax0.imshow(unet.X_val[idx], cmap=cmap0)
                ax0.set_title("image")
                ax0.set_xlabel("pixels")
                ax0.set_ylabel("pixels")
                fig.colorbar(
                    cm.ScalarMappable(cmap=cmap0), ax=ax0, shrink=shrink)
        
                ax1.imshow(unet.y_val[idx], cmap=cmap1)
                ax1.set_title("mask")
                ax1.set_xlabel("pixels")
                fig.colorbar(
                    cm.ScalarMappable(cmap=cmap1), ax=ax1, shrink=shrink)
                
                ax2.imshow(prds[idx], cmap=cmap2)
                ax2.set_title("prediction")
                ax2.set_xlabel("pixels")
                fig.colorbar(
                    cm.ScalarMappable(cmap=cmap2), ax=ax2, shrink=shrink)
                
                # Save
                plt.tight_layout()
                plt.savefig(
                    self.unet.save_path / f"predict_example_{i:02d}.png",
                    format="png"
                    )
                plt.close(fig)
            
        def plot(self):
                   
            # Fetch
            epochs = self.epochs
            trn_losses = self.trn_losses
            val_losses = self.val_losses
            best_epoch = self.best_epoch
            best_val_loss = self.best_val_loss
            best_epoch_time = self.epoch_times[best_epoch]
            save_name = self.unet.save_name
            
            # Info
            infos = (
                f"input shape      : "
                f"{self.unet.X_trn.shape[0]}x" 
                f"{self.unet.X_trn.shape[1]}x"
                f"{self.unet.X_trn.shape[2]}\n"
                f"downscale steps  : {self.unet.downscale_steps}\n"
                f"backbone         : {self.unet.backbone}\n"
                f"batch size       : {self.unet.batch_size}\n"
                f"validation_split : {self.unet.validation_split}\n"
                f"learning rate    : {self.unet.learning_rate}\n"
                )
            
            # Plot
            fig, axis = plt.subplots(1, 1, figsize=(6, 6))   
            axis.plot(trn_losses, label="loss")
            axis.plot(val_losses, label="val_loss")
            axis.axvline(
                x=best_epoch, color="k", linestyle=":", linewidth=1)
            axis.axhline(
                y=best_val_loss, color="k", linestyle=":", linewidth=1)
            axis.text(
                best_epoch / epochs, 1.05, f"{best_epoch_time:.2f}s", 
                size=10, color="k",
                transform=axis.transAxes, ha="center", va="center",
                )
            axis.text(
                1.05, best_val_loss, f"{best_val_loss:.4f}", 
                size=10, color="k",
                transform=axis.transAxes, ha="left", va="center",
                )
            axis.text(
                0.08, 0.85, infos, 
                size=8, color="k",
                transform=axis.transAxes, ha="left", va="top", 
                fontfamily="Consolas",
                )
            axis.set_title(save_name)
            axis.set_xlim(0, epochs)
            axis.set_ylim(0, 1)
            axis.set_xlabel("epochs")
            axis.set_ylabel("loss")
            axis.legend(
                loc="upper left", frameon=False, 
                bbox_to_anchor=(0.05, 0.975), 
                )
            
            # Save    
            plt.tight_layout()
            plt.savefig(self.unet.save_path / "train_plot.png", format="png")
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            self.epoch_t0 = time.time()
    
        def on_epoch_end(self, epoch, logs=None):
            epoch_duration = time.time() - self.epoch_t0
            self.epoch_durations.append(epoch_duration)
            self.epoch_times.append(np.sum(self.epoch_durations))
            self.trn_losses.append(logs.get("loss"))
            self.val_losses.append(logs.get("val_loss"))
            self.trn_metrics.append(logs.get(self.unet.metric))
            self.val_metrics.append(logs.get("val_" + self.unet.metric))
            self.best_epoch = np.argmin(self.val_losses)
            self.best_val_loss = np.min(self.val_losses)
            self.print_log()
            
        def on_train_end(self, logs=None):
            self.predict()
            self.plot()
            
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
        epochs=10,
        downscale_steps=3,
        )
    X = unet.X
    y = unet.y
    unet.train()
    
    # val_prds = unet.val_prds

    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(X_val)
    # viewer.add_image(y_val) 
    