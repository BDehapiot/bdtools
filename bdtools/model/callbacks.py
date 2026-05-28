#%% Imports -------------------------------------------------------------------

import time
import tifffile
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Tensorflow
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

# Scikit learn
from sklearn.metrics import confusion_matrix, classification_report

#%% Class(CallBacks) ----------------------------------------------------------

class CallBacks(Callback):
    
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.X_trn, self.y_trn = self.main.X_trn, self.main.y_trn
        self.X_val, self.y_val = self.main.X_val, self.main.y_val
        self.parameters = main.parameters
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
            filepath=self.main.model_path / "weights.keras",
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
        if self.main.model_type in ["sm", "aec"]:
            self.predict_images()
        elif self.main.model_type == "cls":
            self.predict_classes()
        
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
        
        if self.model_type == "sm":
            var_str = (
                f"backbone         : {self.backbone}\n"
                )
            
        if self.model_type == "cls":
            var_str = (
                f"filters          : {self.filters }\n"
                f"n_class          : {self.n_classes}\n"
                )
        
        if self.model_type == "aec":
            var_str = (
                f"filters          : {self.filters }\n"
                f"latent size      : {self.latent_size}\n"
                )
        
        infos = (
            f"input shape      : "
                f"{'x'.join(str(s) for s in self.main.X_trn.shape)}\n"
            f"{var_str}"
            f"batch size       : {self.batch_size}\n"
            f"validation_split : {self.validation_split}\n"
            f"learning rate    : {self.learning_rate}\n"
            f"best_val_loss    : {best_val_loss:.4f} ({self.loss})\n"
            f"best_val_metric  : {best_val_metric:.4f} ({self.metric})\n"
            )
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))   
        ax.plot(trn_losses, label="loss")
        ax.plot(val_losses, label="val_loss")
        ax.axvline(x=best_epoch, color="k", linestyle=":", linewidth=1)
        ax.axhline(y=best_val_loss, color="k", linestyle=":", linewidth=1)
        ax.text(
            best_epoch / epochs, 1.025, f"{best_epoch_time:.2f}s", 
            size=10, color="k",
            transform=ax.transAxes, ha="center", va="center",
            )
        y_min, y_max = ax.get_ylim()
        bvl_position = (best_val_loss - y_min) / (y_max - y_min)
        ax.text(
            1.025, bvl_position, f"{best_val_loss:.4f}", 
            size=10, color="k",
            transform=ax.transAxes, ha="left", va="center",
            )
        ax.text(
            0.075, 0.85, infos, 
            size=8, color="k",
            transform=ax.transAxes, ha="left", va="top", 
            fontfamily="Consolas",
            )
        
        ax.set_title(model_name, pad=20)
        ax.set_xlim(0, epochs)
        ax.set_ylim(0, np.mean(val_losses) * 2)
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss")
        ax.legend(
            loc="upper left", frameon=False, 
            bbox_to_anchor=(0.05, 0.975), 
            )
        
        # Save    
        plt.tight_layout()
        plt.savefig(self.main.model_path / "train_plot.png", format="png")
        plt.show()
        
#%% Class(CallBacks) : predict_images() --------------------------------------- 
        
    def predict_images(self, max_img=50, max_mb=100):
                 
        # Determine size
        nS = self.X_val.shape[0]
        nY = self.X_val.shape[1]
        nX = self.X_val.shape[2]
        nC = self.input_shape[-1]
        max_size = (((max_mb * 2 ** 20) / (nY * nX)) / 4 ) * nC
        max_size = np.floor(max_size).astype(int)
        size = np.min([max_img, max_size, nS])
        
        # Predict
        idxs = np.random.choice(
            self.X_val.shape[0], size=size, replace=False)
        prds = self.main.predict(self.X_val[idxs, ...]).squeeze()
        
        # Assemble display
        examples = []
        for i, idx in enumerate(idxs):
            
            img = self.X_val[idx]
            prd = prds[i]
            if self.model_type == "sm":
                gtr = self.y_val[idx]
            if self.model_type == "aec":
                gtr = self.X_val[idx]
                
            if img.ndim > 2:
                if gtr.ndim == 2:
                    gtr = np.tile(gtr[:, :, np.newaxis], (1, 1, nC))
                if prd.ndim == 2:
                    prd = np.tile(prd[:, :, np.newaxis], (1, 1, nC))
        
            examples.append(np.hstack((img, gtr, prd)))
        examples = np.stack(examples)  
        
        # Add borders
        for i in range(2):
            examples[:, :, (nX * (i + 1)) - 1, ...] = 1
        examples = (examples * 255).astype("uint8")
        
        # Save
        if nC == 1:
            axes = "ZYX"
        else:
            axes = "ZCYX"
            examples = np.transpose(examples, (0, 3, 1, 2))
        tifffile.imwrite(
            self.main.model_path / "predict_images.tif",
            examples, imagej=True, metadata={"axes": axes},
            )
        
#%% Class(CallBacks) : predict_classes() --------------------------------------

    def predict_classes(self):
        
        # Predict
        prds = self.main.predict(self.X_val).squeeze()
        y_pred = np.argmax(prds, axis=1)
        y_true = np.argmax(self.y_val, axis=1)
        
        # Confusion matrix & statistics
        cmat = confusion_matrix(y_true, y_pred)
        stat = classification_report(y_true, y_pred)
        
        # infos
        infos = []
        infos.append(" cls    prc     rec     f1s  ")
        infos.append("----------------------------")
        for c in range(self.n_classes):
            if self.classes is not None:
                cls_str = f"({ self.classes[c]})"
            else:
                cls_str = ""
            infos.append(
                f" {c:03d} | "
                f"{stat[str(c)]['precision']:.3f} | "
                f"{stat[str(c)]['recall'   ]:.3f} | "
                f"{stat[str(c)]['f1-score' ]:.3f} "
                f"{cls_str}"
                )
        infos.append("----------------------------")
        infos.append(
            " avg | "
            f"{stat['macro avg']['precision']:.3f} | "
            f"{stat['macro avg']['recall'   ]:.3f} | "
            f"{stat['macro avg']['f1-score' ]:.3f}"
            )
        infos.append(
            "wavg | "
            f"{stat['weighted avg']['precision']:.3f} | "
            f"{stat['weighted avg']['recall'   ]:.3f} | "
            f"{stat['weighted avg']['f1-score' ]:.3f}"
            )
        infos = "\n".join(infos)
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 6)) 
        img = ax.imshow(cmat, interpolation="nearest")
        ax.figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        
        ax.text(
            0.0, -0.25, infos, 
            size=12, color="k", font="consolas",
            transform=ax.transAxes, ha="left", va="top",
            )

        ax.set_title(self.model_name)
        ax.set_xticks(np.arange(cmat.shape[1]))
        ax.set_yticks(np.arange(cmat.shape[0]))
        ax.set_xticklabels(self.classes, rotation=90)
        ax.set_yticklabels(self.classes)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")      
        
        # Save    
        plt.tight_layout()
        plt.savefig(self.main.model_path / "predict_classes.png", format="png")
        plt.show()