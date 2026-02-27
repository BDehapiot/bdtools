#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Tensorflow
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

#%% Class(CallBacks) ----------------------------------------------------------

class CallBacks(Callback):
    
    def __init__(self, unet):
        super().__init__()
        
        # Fetch
        self.unet = unet
        
        # Execute
        self.initialize()
        
#%% Class(CallBacks) : initialize() -------------------------------------------
        
    def initialize(self):

        self.trn_losses  = []
        self.val_losses  = []
        self.trn_metrics = []
        self.val_metrics = []
        self.epoch_times = []
        self.epoch_durations = []
        
        # Checkpoint
        self.checkpoint = ModelCheckpoint(
            filepath=Path(self.unet.model_path, "weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss", 
            mode="min",
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.unet.patience, 
            monitor="val_loss",
            mode="min",
            )
        
    def set_model(self, model):
        self.model = model
        self.checkpoint.set_model(model)
        self.early_stopping.set_model(model)
               
#%% Class(CallBacks) : events -------------------------------------------------    
        
    def on_train_begin(self, logs=None):
        self.checkpoint.on_train_begin(logs)
        self.early_stopping.on_train_begin(logs)
        
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
        self.checkpoint.on_epoch_end(epoch, logs)
        self.early_stopping.on_epoch_end(epoch, logs)
        self.print_log()
        
    def on_train_end(self, logs=None):
        self.checkpoint.on_train_end(logs)
        self.early_stopping.on_train_end(logs)
        self.plot_training()
        self.predict_examples()
        
#%% Class(CallBacks) : print_log() --------------------------------------------    
        
    def print_log(self):
        
        # Fetch
        epoch = self.epoch
        epochs = self.unet.epochs - 1
        trn_loss = self.trn_losses[-1]
        val_loss = self.val_losses[-1]
        best_val_loss = self.best_val_loss
        trn_metric = self.trn_metrics[-1]
        val_metric = self.val_metrics[-1]
        wait = self.early_stopping.wait
        patience = self.unet.patience

        # Print
        print(
            f"epoch {epoch:>{len(str(epochs))}}/{epochs} "
            f"wait {wait:>{len(str(patience))}}/{patience} "
            f"({best_val_loss:.4f}) "
            f"l|{trn_loss:.4f}| "
            f"vl({val_loss:.4f}) "
            f"m|{trn_metric:.4f}| "
            f"vm({val_metric:.4f}) "
            )

#%% Class(CallBacks) : plot_training() ----------------------------------------    
        
    def plot_training(self):
               
        # Fetch
        epochs = len(self.trn_losses)
        trn_losses = self.trn_losses
        val_losses = self.val_losses
        best_epoch = self.best_epoch
        best_epoch_time = self.epoch_times[best_epoch]
        best_val_loss = self.best_val_loss
        best_val_metric = self.val_metrics[best_epoch]
        metric = self.unet.metric
        model_name = self.unet.model_name
        
        # Info
        infos = (
            f"input shape      : "
                f"{'x'.join(str(s) for s in self.unet.X_trn.shape)}\n"
            f"backbone         : {self.unet.backbone}\n"
            f"batch size       : {self.unet.batch_size}\n"
            f"validation_split : {self.unet.validation_split}\n"
            f"learning rate    : {self.unet.learning_rate}\n"
            f"best_val_loss    : {best_val_loss:.4f}\n"
            f"best_val_metric  : {best_val_metric:.4f} ({metric})\n"
            )
        
        # Plot
        fig, axis = plt.subplots(1, 1, figsize=(6, 6))   
        axis.plot(trn_losses, label="loss")
        axis.plot(val_losses, label="val_loss")
        axis.axvline(x=best_epoch, color="k", linestyle=":", linewidth=1)
        axis.axhline(y=best_val_loss, color="k", linestyle=":", linewidth=1)
        axis.text(
            best_epoch / epochs, 1.025, f"{best_epoch_time:.2f}s", 
            size=10, color="k",
            transform=axis.transAxes, ha="center", va="center",
            )
        axis.text(
            1.025, best_val_loss, f"{best_val_loss:.4f}", 
            size=10, color="k",
            transform=axis.transAxes, ha="left", va="center",
            )
        axis.text(
            0.075, 0.85, infos, 
            size=8, color="k",
            transform=axis.transAxes, ha="left", va="top", 
            fontfamily="Consolas",
            )
        
        axis.set_title(model_name, pad=20)
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
        plt.savefig(self.unet.model_path / "train_plot.png", format="png")
        plt.show()
        
#%% Class(CallBacks) : predict_examples() ------------------------------------- 
        
    def predict_examples(self, max_img=50, max_mb=100):
                 
        # Determine 
        nI, nY, nX = self.unet.X_val.shape
        max_size = ((max_mb * 2**20) / (nY * nX)) / 4 
        max_size = np.floor(max_size).astype(int)
        size = np.min([max_img, max_size, nI])
        
        # Predict
        idxs = np.random.choice(
            self.unet.X_val.shape[0], size=size, replace=False)
        prds = self.model.predict(self.unet.X_val[idxs, ...]).squeeze()
                
        # Assemble predict_examples
        predict_examples = []
        for i, idx in enumerate(idxs):
            img = self.unet.X_val[idx]
            gtr = self.unet.y_val[idx]
            prd = prds[i].squeeze()
            acc = np.abs(gtr - prd)
            predict_examples.append(
                np.hstack((img, gtr, prd, acc))
                )
        predict_examples = np.stack(predict_examples)  
        for i in range(3):
            width = prds[i].squeeze().shape[1]
            predict_examples[:, :, width * (i + 1)] = 1
        
        # Save
        io.imsave(
            self.unet.model_path / "predict_examples.tif",
            predict_examples.astype("float32"), check_contrast=False
            )
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import napari
    from skimage.measure import label
    from bdtools.norm import norm_pct
    from bdtools.model.unet import UNet
    
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
    X_trn = unet.X_trn
    y_trn = unet.y_trn
    X_val = unet.X_val
    y_val = unet.y_val
    
#%%

    max_img=50
    max_mb=100
    
    # Determine 
    nI, nY, nX = unet.X_val.shape
    max_size = ((max_mb * 2**20) / (nY * nX)) / 4 
    max_size = np.floor(max_size).astype(int)
    size = np.min([max_img, max_size, nI])

    # Predict
    idxs = np.random.choice(
        unet.X_val.shape[0], size=size, replace=False)