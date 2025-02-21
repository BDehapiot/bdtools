#%% Imports -------------------------------------------------------------------

import json
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm

# bdtools
import metrics

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

# Skimage
from skimage.transform import downscale_local_mean, resize

# Matplotlib
import matplotlib.pyplot as plt

#%% Function(s) ---------------------------------------------------------------

def downscale(X, y=None, steps=1):
    df = 2**steps
    X = downscale_local_mean(X, (1, df, df))
    if y is not None:
        y = downscale_local_mean(y, (1, df, df))
        return X, y
    else:
        return X
    
def upscale(X, shape):
    return resize(X, shape, order=1)
    
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
            downscale_steps=0, 
            ):
        
        # Fetch
        self.save_name = save_name
        self.load_name = load_name
        self.root_path = root_path
        
        # Paths
        if self.save_name: self.model_name = f"model_{save_name}"
        if self.load_name: self.model_name = f"model_{load_name}"
        self.model_path = self.root_path / self.model_name
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)
        
        # build_params
        if self.load_name:
            with open(str(self.model_path / "build_params.pkl"), "rb") as file:
                self.build_params = pickle.load(file)
        else:
            self.build_params = {
                "classes" : classes,
                "backbone" : backbone,
                "activation" : activation,
                "downscale_steps" : downscale_steps,
                }

        # Execute
        self.build()
        
#%% Build ---------------------------------------------------------------------

    def build(self):
        
        # Fetch
        self.backbone = self.build_params["backbone"]
        self.classes = self.build_params["classes"]
        self.activation = self.build_params["activation"]
        self.downscale_steps = self.build_params["downscale_steps"]

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
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            metric="soft_dice_coef",
            learning_rate=0.001,
            patience=20,
            ):

        # Fetch
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.metric = metric
        self.learning_rate = learning_rate
        self.patience = patience

        # Prepare data
        if X_val is None:
            self.X_trn, self.y_trn, self.X_val, self.y_val = split(
                X, y, split=validation_split)
        else:
            self.X_trn, self.y_trn = X, y
            self.X_val, self.y_val = X_val, y_val

        # Downscale
        if self.downscale_steps > 0:
            X_trn, y_trn = downscale(
                self.X_trn, y=self.y_trn, steps=self.downscale_steps)
            X_val, y_val = downscale(
                self.X_val, y=self.y_val, steps=self.downscale_steps)

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
                x=X_trn, y=y_trn,
                validation_data=(X_val, y_val),
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
        
        # Save train_params
        with open(str(self.model_path / "train_params.pkl"), "wb") as file:
            pickle.dump(self.train_params, file)  
        with open(str(self.model_path / "train_params.txt"), "w") as file:
            json.dump(self.train_params, file, indent=4)
        
#%% Predict -------------------------------------------------------------------

    def predict(self, X):
        
        # Downscale
        if self.downscale_steps > 0:
            shape = X.shape
            X = downscale(X, steps=self.downscale_steps)

        # Predict
        prds = self.model.predict(X).squeeze()
        
        # Upscale
        if self.downscale_steps > 0:
            prds = upscale(prds, shape)

        return prds

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
            f"vl({val_loss:.4f}) "
            f"l|{trn_loss:.4f}| "
            f"vm|{val_metric:.4f}| "
            f"m|{trn_metric:.4f}| "
            )
        
    def plot_training(self):
               
        # Fetch
        epochs = self.unet.epochs
        trn_losses = self.trn_losses
        val_losses = self.val_losses
        best_epoch = self.best_epoch
        best_epoch_time = self.epoch_times[best_epoch]
        best_val_loss = self.best_val_loss
        best_val_metric = self.val_metrics[best_epoch]
        metric = self.unet.metric
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
            f"best_val_loss    : {best_val_loss:.4f}\n"
            f"best_val_metric  : {best_val_metric:.4f} ({metric})\n"
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
        
        axis.set_title(save_name)
        axis.set_xlim(0, len(self.trn_losses))
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
        idxs = np.random.randint(0, unet.X_val.shape[0], size=size) 
        prds = self.unet.predict(unet.X_val[idxs, ...].squeeze())
                
        # Assemble predict_examples
        predict_examples = []
        for i, idx in enumerate(idxs):
            img = unet.X_val[idx]
            gtr = unet.y_val[idx]
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
            unet.model_path / "predict_examples.tif",
            predict_examples.astype("float32"), check_contrast=False
            )
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path
    from bdtools.patch import merge_patches   
    from bdtools.models import preprocess, augment

    # Parameters
    dataset = "em_mito"
    # dataset = "fluo_nuclei"
    patch_size = 256
    img_norm = "global"
    msk_type = "edt"
    
    # Paths
    local_path = Path.cwd().parent.parent / "_local"
    X_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
    y_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    X_val_path = local_path / f"{dataset}" / f"{dataset}_val.tif"
    y_val_path = local_path / f"{dataset}" / f"{dataset}_msk_val.tif"
    
    # Load images & masks
    X = io.imread(X_path)
    y = io.imread(y_path)
    X_val = io.imread(X_val_path)
    y_val = io.imread(y_val_path)
    
    # Model (training procedure) ----------------------------------------------
    
    # Preprocess
    t0 = time.time()
    print("preprocess :", end=" ", flush=True)
    X, y = preprocess(
        X, msks=y, 
        img_norm=img_norm, 
        msk_type=msk_type, 
        patch_size=patch_size
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Preprocess
    t0 = time.time()
    print("preprocess :", end=" ", flush=True)
    X_val, y_val = preprocess(
        X_val, msks=y_val, 
        img_norm=img_norm, 
        msk_type=msk_type, 
        patch_size=patch_size
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Augment
    t0 = time.time()
    print("augment :", end=" ", flush=True)
    X, y = augment(X, y, 5000,
        gamma_p=0.0, gblur_p=0.0, noise_p=0.0, flip_p=0.5, distord_p=0.5)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # # # Display
    # # viewer = napari.Viewer()
    # # viewer.add_image(X, contrast_limits=[0, 1])
    # # viewer.add_image(y) 
    
    unet = UNet(
        save_name=f"test_{dataset}",
        load_name="",
        root_path=Path.cwd(),
        backbone="resnet18",
        classes=1,
        activation="sigmoid",
        downscale_steps=1, 
        )
    
    unet.train(
        X, y, 
        # X_val=None, y_val=None,
        X_val=X_val, y_val=y_val,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        metric="soft_dice_coef",
        learning_rate=0.0005,
        patience=20,
        )
    
    # Model (predict procedure) -----------------------------------------------
    
    # # Preprocess
    # t0 = time.time()
    # print("preprocess :", end=" ", flush=True)
    # X_val_prep = preprocess(
    #     X_val, msks=None, 
    #     img_norm=img_norm,
    #     msk_type=msk_type,
    #     patch_size=patch_size,
    #     patch_overlap=patch_size // 2,
    #     )
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # unet = UNet(load_name=f"test_{dataset}")
    # prds = unet.predict(X_val_prep)
    # prds = merge_patches(prds, X_val.shape, patch_size // 2)
            
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(X_val)
    # viewer.add_image(prds) 
    