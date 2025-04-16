#%% Imports -------------------------------------------------------------------

import sys
import time
import json
import napari
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm

# bdtools
from bdtools.norm import norm_pct
from bdtools.patch import merge_patches 
from bdtools.models import preprocess, augment, metrics

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

# Skimage
from skimage.transform import rescale, resize

# Matplotlib
import matplotlib.pyplot as plt

#%% Comments

#%% Function(s) ---------------------------------------------------------------

def downscale(X, y=None, df=2):
    rf = 1 / df
    X = rescale(X, (1, rf, rf), preserve_range=True, order=0)
    if y is not None:
        y = rescale(y, (1, rf, rf), preserve_range=True, order=0)
        return X, y
    else:
        return X
    
def upscale(X, shape):
    return resize(X, shape, order=0)

def closest_32(n):
    return round(n / 32) * 32
    
def split(X, y, split=0.2):
    n_total = X.shape[0]
    n_val = int(n_total * split)
    idx = np.random.permutation(np.arange(0, n_total))
    X_trn = X[idx[n_val:]] 
    y_trn = y[idx[n_val:]]
    X_val = X[idx[:n_val]]
    y_val = y[idx[:n_val]]
    return X_trn, y_trn, X_val, y_val

#%% UNet() --------------------------------------------------------------------

class UNet:
       
    def __init__(           
            self,
            save_name="",
            load_name="",
            root_path=Path.cwd(),
            backbone="resnet18",
            classes=1,
            activation="sigmoid",
            ):
        
        # Fetch
        self.save_name = save_name
        self.load_name = load_name
        self.root_path = root_path
        
        # Paths
        if self.load_name: 
            self.model_path = self.root_path / self.load_name
        
        # build_params
        if self.load_name:
            with open(str(self.model_path / "build_params.pkl"), "rb") as file:
                self.build_params = pickle.load(file)
            with open(str(self.model_path / "preprocess_params.pkl"), "rb") as file:
                self.preprocess_params = pickle.load(file)
            with open(str(self.model_path / "train_params.pkl"), "rb") as file:
                self.train_params = pickle.load(file)
        else:
            self.build_params = {
                "classes" : classes,
                "backbone" : backbone,
                "activation" : activation,
                }

        # Execute
        self.build()
        
#%% Build ---------------------------------------------------------------------

    def build(self):
        
        # Fetch
        self.backbone = self.build_params["backbone"]
        self.classes = self.build_params["classes"]
        self.activation = self.build_params["activation"]

        # Build
        self.model = sm.Unet(
            self.build_params["backbone"], 
            input_shape=(None, None, 1), # Parameter
            classes=self.build_params["classes"],
            activation=self.build_params["activation"],
            encoder_weights=None,
            )
        
        # Load weights
        if self.load_name:
            self.model.load_weights(Path(self.model_path, "weights.h5"))
            
#%% Train ---------------------------------------------------------------------

    def train(
            
            self, 
            X, y, 
            X_val=None, y_val=None,
            preview=False,
            
            # Preprocess
            img_norm="global", 
            msk_type="normal", 
            patch_size=256,
            patch_overlap=0,
            downscaling_factor=1,

            # Augment
            iterations=0,
            invert_p=0.5,
            gamma_p=0.5, 
            gblur_p=0.5, 
            noise_p=0.5, 
            flip_p=0.5, 
            distord_p=0.5,
            
            # Train
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            metric="soft_dice_coef",
            learning_rate=0.001,
            patience=20,
            
            ):

        # Fetch
        self.preview = preview
        self.img_norm = img_norm
        self.msk_type = msk_type 
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.downscaling_factor = downscaling_factor 
        self.iterations = iterations
        self.invert_p = invert_p
        self.gamma_p = gamma_p
        self.gblur_p = gblur_p
        self.noise_p = noise_p
        self.flip_p = flip_p
        self.distord_p = distord_p
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.metric = metric
        self.learning_rate = learning_rate
        self.patience = patience    
                        
        # Split data
        if X_val is None:
            self.X_trn, self.y_trn, self.X_val, self.y_val = split(
                X, y, split=validation_split)
        else:
            self.X_trn, self.y_trn = X, y
            self.X_val, self.y_val = X_val, y_val
        
        # Downscale data
        if self.downscaling_factor > 1:
            t0 = time.time()
            print("train - downscale data : ", end="", flush=True)
            self.X_trn, self.y_trn = downscale(
                self.X_trn, y=self.y_trn, df=self.downscaling_factor)
            self.X_val, self.y_val = downscale(
                self.X_val, y=self.y_val, df=self.downscaling_factor)
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")   

        # Preprocess data
        t0 = time.time()
        print("train - preprocess data : ", end="", flush=True)
        self.X_trn_prp, self.y_trn_prp = preprocess(
            self.X_trn, msks=self.y_trn, 
            img_norm=self.img_norm, 
            msk_type=self.msk_type, 
            patch_size=closest_32(
                self.patch_size // self.downscaling_factor),
            patch_overlap=self.patch_overlap // self.downscaling_factor
            )
        self.X_val_prp, self.y_val_prp = preprocess(
            self.X_val, msks=self.y_val, 
            img_norm=self.img_norm, 
            msk_type=self.msk_type, 
            patch_size=closest_32(
                self.patch_size // self.downscaling_factor),
            patch_overlap=self.patch_overlap // self.downscaling_factor
            )
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")

        # Augment data
        self.X_trn_prp_n = self.X_trn_prp.shape[0]
        if self.iterations > 0:
            t0 = time.time()
            print("train - augment data : ", end="", flush=True)
            self.X_trn_prp, self.y_trn_prp = augment(
                self.X_trn_prp, self.y_trn_prp, self.iterations,
                invert_p=self.invert_p, 
                gamma_p=self.gamma_p, 
                gblur_p=self.gblur_p, 
                noise_p=self.noise_p, 
                flip_p=self.flip_p, 
                distord_p=self.distord_p,
                )
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
           
        # Display preview
        if self.preview:
            viewer = napari.Viewer()
            viewer.add_image(self.X_trn_prp)
            viewer.add_image(self.y_trn_prp)
            sys.exit()
            
        # Paths
        if self.save_name: 
            self.model_name = f"model_{self.save_name}"
        else:
            self.model_name = (
                "model_"
                f"{self.patch_size}_"
                f"{self.msk_type}_"
                f"{self.iterations}-{self.X_trn_prp_n}_"
                f"{self.downscaling_factor}"
                )
        self.model_path = self.root_path / self.model_name
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)
                        
        # preprocess_params
        self.preprocess_params ={
            "img_norm" : self.img_norm, 
            "msk_type" : self.msk_type, 
            "patch_size" : self.patch_size,
            "patch_overlap" : self.patch_overlap,
            "downscaling_factor" : self.downscaling_factor, 
            }
        
        # augment_params
        self.augment_params ={
            "iterations" : self.iterations, 
            "gamma_p" : self.gamma_p, 
            "gblur_p" : self.gblur_p,
            "noise_p" : self.noise_p,
            "flip_p" : self.flip_p, 
            "distord_p" : self.distord_p,
            }

        # train_params
        self.train_params ={
            "epochs" : self.epochs,
            "batch_size" : self.batch_size,
            "validation_split" : self.validation_split,
            "metric" : self.metric,
            "learning_rate" : self.learning_rate,
            "patience" : self.patience,
            }

        # Save build_params
        with open(str(self.model_path / "build_params.pkl"), "wb") as file:
            pickle.dump(self.build_params, file) 
        with open(str(self.model_path / "build_params.txt"), "w") as file:
            json.dump(self.build_params, file, indent=4)
            
        # Save preprocess_params
        with open(str(self.model_path / "preprocess_params.pkl"), "wb") as file:
            pickle.dump(self.preprocess_params, file) 
        with open(str(self.model_path / "preprocess_params.txt"), "w") as file:
            json.dump(self.preprocess_params, file, indent=4)
            
        # Save augment_params
        with open(str(self.model_path / "augment_params.pkl"), "wb") as file:
            pickle.dump(self.augment_params, file) 
        with open(str(self.model_path / "augment_params.txt"), "w") as file:
            json.dump(self.augment_params, file, indent=4)
        
        # Save train_params
        with open(str(self.model_path / "train_params.pkl"), "wb") as file:
            pickle.dump(self.train_params, file)  
        with open(str(self.model_path / "train_params.txt"), "w") as file:
            json.dump(self.train_params, file, indent=4)

        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy", # Parameter
            metrics=[getattr(metrics, metric)],
            )
        
        # Callbacks
        self.callbacks = [CallBacks(self)]
        
        try:
        
            # Train
            self.history = self.model.fit(
                x=self.X_trn_prp, y=self.y_trn_prp,
                validation_data=(self.X_val_prp, self.y_val_prp),
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
        
#%% Predict -------------------------------------------------------------------

    def predict(self, X, verbose=True):

        def log(message, end="\n"):
            if verbose > 2:
                print(message, end=end, flush=True)
    
        # Fetch parameters
        patch_size = self.preprocess_params["patch_size"]
        downscaling_factor = self.preprocess_params["downscaling_factor"]
    
        # Adjust variables
        patch_size = closest_32(patch_size // downscaling_factor)
    
        # Downscale
        if downscaling_factor > 1:
            t0 = time.time()
            log("predict - downscale data : ", end="")
            shape = X.shape
            X = downscale(X, df=downscaling_factor)
            shape_dsc = X.shape
            t1 = time.time()
            log(f"{t1 - t0:.3f}s")
        
        # Preprocess
        t0 = time.time()
        log("predict - preprocess data : ", end="")
        X_prp = preprocess(
            X, msks=None, 
            img_norm=self.preprocess_params["img_norm"], 
            patch_size=patch_size,
            patch_overlap=patch_size // 2,
        )
        t1 = time.time()
        log(f"{t1 - t0:.3f}s")
    
        # Predict
        t0 = time.time()
        log("predict - predict : ", end="")
        prds = self.model.predict(X_prp, verbose=verbose).squeeze()
        t1 = time.time()
        log(f"{t1 - t0:.3f}s")    
        
        # Merge patches
        t0 = time.time()
        log("predict - merge data : ", end="")
        prds = merge_patches(prds, shape_dsc, patch_size // 2)
        t1 = time.time()
        log(f"{t1 - t0:.3f}s")
        
        # Upscale
        t0 = time.time()
        log("predict - upscale data : ", end="")
        if downscaling_factor > 1:
            prds = upscale(prds, shape)
        t1 = time.time()
        log(f"{t1 - t0:.3f}s")
    
        return prds

    # def predict(self, X):
        
    #     # Fetch
    #     patch_size = self.preprocess_params["patch_size"]
    #     downscaling_factor = self.preprocess_params["downscaling_factor"]
        
    #     # Adjust variables
    #     patch_size = closest_32(patch_size // downscaling_factor)
        
    #     # Downscale
    #     if downscaling_factor > 1:
    #         t0 = time.time()
    #         print("predict - downscale data : ", end="", flush=True)
    #         shape = X.shape
    #         X = downscale(X, df=downscaling_factor)
    #         shape_dsc = X.shape
    #         t1 = time.time()
    #         print(f"{t1 - t0:.3f}s")
        
    #     # Preprocess
    #     t0 = time.time()
    #     print("predict - preprocess data : ", end="", flush=True)
    #     X_prp = preprocess(
    #         X, msks=None, 
    #         img_norm=self.preprocess_params["img_norm"], 
    #         patch_size=patch_size,
    #         patch_overlap=patch_size // 2,
    #         )
    #     t1 = time.time()
    #     print(f"{t1 - t0:.3f}s")

    #     # Predict
    #     prds = self.model.predict(X_prp).squeeze()
            
    #     # Merge patches
    #     t0 = time.time()
    #     print("predict - merge data : ", end="", flush=True)
    #     prds = merge_patches(prds, shape_dsc, patch_size // 2)
    #     t1 = time.time()
    #     print(f"{t1 - t0:.3f}s")
        
    #     # Upscale
    #     t0 = time.time()
    #     print("predict - upscale data : ", end="", flush=True)
    #     if downscaling_factor > 1:
    #         prds = upscale(prds, shape)
    #     t1 = time.time()
    #     print(f"{t1 - t0:.3f}s")

    #     return prds

#%% Callbacks -----------------------------------------------------------------

class CallBacks(Callback):
    
    def __init__(self, unet):
        super().__init__()
        
        # Fetch
        self.unet = unet
        
        # Initialize
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
            monitor='val_loss',
            mode="min",
            )
               
    def set_model(self, model):
        self.model = model
        self.checkpoint.set_model(model)
        self.early_stopping.set_model(model)
        
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
            f"downscaling factor : {self.unet.downscaling_factor}\n"
            f"augmentation       : "
            f"{self.unet.iterations}/{self.unet.X_trn_prp_n}\n"
            f"input shape        : "
            f"{self.unet.X_trn_prp.shape[0]}x" 
            f"{self.unet.X_trn_prp.shape[1]}x"
            f"{self.unet.X_trn_prp.shape[2]}\n"
            f"backbone           : {self.unet.backbone}\n"
            f"batch size         : {self.unet.batch_size}\n"
            f"validation_split   : {self.unet.validation_split}\n"
            f"learning rate      : {self.unet.learning_rate}\n"
            f"best_val_loss      : {best_val_loss:.4f}\n"
            f"best_val_metric    : {best_val_metric:.4f} ({metric})\n"
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
            0.08, 0.85, infos, 
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
        
    def predict_examples(self, size=50):
                    
        # Predict
        idxs = np.random.randint(0, self.unet.X_val_prp.shape[0], size=size) 
        prds = self.model.predict(self.unet.X_val_prp[idxs, ...]).squeeze()
                
        # Assemble predict_examples
        predict_examples = []
        for i, idx in enumerate(idxs):
            img = norm_pct(self.unet.X_val_prp[idx])
            gtr = norm_pct(self.unet.y_val_prp[idx])
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
     
    # Parameters
    dataset = "em_mito"
    # dataset = "fluo_nuclei"
    # dataset = "sat_buildings"
    
    # Paths
    local_path = Path.cwd().parent.parent / "_local"
    X_trn_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
    y_trn_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    X_val_path = local_path / f"{dataset}" / f"{dataset}_val.tif"
    y_val_path = local_path / f"{dataset}" / f"{dataset}_msk_val.tif"
    
    # Load images & masks
    X_trn = io.imread(X_trn_path)
    y_trn = io.imread(y_trn_path)
    X_val = io.imread(X_val_path)
    y_val = io.imread(y_val_path)
    
    # Preprocessing -----------------------------------------------------------

    if dataset == "sat_buildings":
        
        # RGB to grayscale
        X_trn = np.mean(X_trn, axis=-1).astype("uint8")
        X_val = np.mean(X_val, axis=-1).astype("uint8")
        
        # Resize
        X_trn = resize(X_trn, (X_trn.shape[0], 320, 320), order=0)
        y_trn = resize(y_trn, (y_trn.shape[0], 320, 320), order=0)
        X_val = resize(X_val, (X_val.shape[0], 320, 320), order=0)
        y_val = resize(y_val, (y_val.shape[0], 320, 320), order=0)

        # # Display
        # viewer = napari.Viewer()
        # viewer.add_image(X_trn)
        # viewer.add_image(y_trn) 
        
    # Model (training procedure) ----------------------------------------------
    
    # unet = UNet(
    #     save_name="",
    #     load_name="",
    #     root_path=Path.cwd(),
    #     backbone="resnet18",
    #     classes=1,
    #     activation="sigmoid",
    #     )
    
    # unet.train(
        
    #     X_trn, y_trn, 
    #     X_val=None, y_val=None,
    #     # X_val=X_val, y_val=y_val,
    #     preview=1,
        
    #     # Preprocess
    #     img_norm="global", 
    #     msk_type="edt", 
    #     patch_size=256,
    #     patch_overlap=0,
    #     downscaling_factor=2, 
        
    #     # Augment
    #     iterations=2000,
    #     invert_p=0.5,
    #     gamma_p=0.5, 
    #     gblur_p=0, 
    #     noise_p=0, 
    #     flip_p=0.5, 
    #     distord_p=0.5,
        
    #     # Train
    #     epochs=100,
    #     batch_size=8,
    #     validation_split=0.2,
    #     metric="soft_dice_coef",
    #     learning_rate=0.0005,
    #     patience=20,
        
    #     )
    
    # Model (predict procedure) -----------------------------------------------
    
    # unet = UNet(
    #     load_name="model_256_edt_2000-1584_2",
    #     )
    # prds = unet.predict(X_val, verbose=3)
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(X_val)
    # viewer.add_image(prds) 
    