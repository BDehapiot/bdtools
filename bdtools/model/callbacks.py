#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Tensorflow
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

#%% Class(CallBacks) ----------------------------------------------------------

class CallBacks(Callback):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.X_trn, self.y_trn = self.model.X_trn, self.model.y_trn
        self.X_val, self.y_val = self.model.X_val, self.model.y_val
        self.parameters = model.parameters
        for key, val in self.parameters.items():
            setattr(self, key, val)
        
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
            filepath=self.model.model_path / "weights.keras",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss", 
            mode="min",
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience, 
            monitor="val_loss",
            mode="min",
            )
        
    def set_model(self, mmodel):
        self.mmodel = mmodel
        self.checkpoint.set_model(mmodel)
        self.early_stopping.set_model(mmodel)
               
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
        self.trn_metrics.append(logs.get(self.metric))
        self.val_metrics.append(logs.get("val_" + self.metric))
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
        epochs = self.epochs - 1
        trn_loss = self.trn_losses[-1]
        val_loss = self.val_losses[-1]
        best_val_loss = self.best_val_loss
        trn_metric = self.trn_metrics[-1]
        val_metric = self.val_metrics[-1]
        wait = self.early_stopping.wait
        patience = self.patience

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
        model_name = self.model_name
        
        # Info
        
        # /////////////////////////////////////////////////////////////////////
        
        if hasattr(self, "backbone"):
            var_str = (
                f"backbone         : {self.backbone}\n"
                )
        elif hasattr(self, "filters"):
            var_str = (
                f"filters          : {self.filters }\n"
                f"latent size      : {self.latent_size}\n"
                )
        
        # /////////////////////////////////////////////////////////////////////
        
        infos = (
            f"input shape      : "
                f"{'x'.join(str(s) for s in self.model.X_trn.shape)}\n"
            f"{var_str}"
            f"batch size       : {self.batch_size}\n"
            f"validation_split : {self.validation_split}\n"
            f"learning rate    : {self.learning_rate}\n"
            f"best_val_loss    : {best_val_loss:.4f} ({self.loss})\n"
            f"best_val_metric  : {best_val_metric:.4f} ({self.metric})\n"
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
        y_min, y_max = axis.get_ylim()
        bvl_position = (best_val_loss - y_min) / (y_max - y_min)
        axis.text(
            1.025, bvl_position, f"{best_val_loss:.4f}", 
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
        axis.set_ylim(0, np.mean(val_losses) * 2)
        axis.set_xlabel("epochs")
        axis.set_ylabel("loss")
        axis.legend(
            loc="upper left", frameon=False, 
            bbox_to_anchor=(0.05, 0.975), 
            )
        
        # Save    
        plt.tight_layout()
        plt.savefig(self.model.model_path / "train_plot.png", format="png")
        plt.show()
        
#%% Class(CallBacks) : predict_examples() ------------------------------------- 
        
    def predict_examples(self, max_img=50, max_mb=100):
                 
        # Determine size
        nS = self.X_val.shape[0]
        nY = self.X_val.shape[1]
        nX = self.X_val.shape[2]
        max_size = ((max_mb * 2 ** 20) / (nY * nX)) / 4 
        max_size = np.floor(max_size).astype(int)
        size = np.min([max_img, max_size, nS])
        
        # Predict
        idxs = np.random.choice(
            self.X_val.shape[0], size=size, replace=False)
        prds = self.model.predict(self.X_val[idxs, ...]).squeeze()
                
        # Assemble predict_examples
        predict_examples = []
        for i, idx in enumerate(idxs):
            img = self.X_val[idx]
            if self.input_shape[-1] > 1:
                img = np.mean(img, axis=-1)
                
            # /////////////////////////////////////////////////////////////////
                
            if hasattr(self, "backbone"):
                gtr = self.y_val[idx]
            elif hasattr(self, "filters"): 
                gtr = self.X_val[idx]
                
            # /////////////////////////////////////////////////////////////////
            
            prd = prds[i].squeeze()
            acc = np.abs(gtr - prd)
            predict_examples.append(
                np.hstack((img, gtr, prd, acc))
                )
        predict_examples = np.stack(predict_examples)  
        for i in range(3):
            width = prds[i].squeeze().shape[1]
            predict_examples[:, :, width * (i + 1)] = 1
        predict_examples = (predict_examples * 255).astype("uint8")
        
        # Save
        io.imsave(
            self.model.model_path / "predict_examples.tif",
            predict_examples, check_contrast=False
            )