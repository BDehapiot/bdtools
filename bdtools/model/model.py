#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from pathlib import Path

# bdtools
from bdtools.check import Check
from bdtools.model.build import Build 
from bdtools.model.prepare import Prepare
from bdtools.model.callbacks import CallBacks
from bdtools.patch import extract_patches, merge_patches

#%% Parameters ----------------------------------------------------------------

parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    "model_type"         : "cls",
    "input_shape"        : (None, None, 3),
    "backbone"           : "resnet18", # (sm)
    "filters"            : [64, 128], # (cls & aec)
    "n_classes"          : 12, # (cls)
    "classes"            : None,
    "latent_size"        : 128, # (aec)
    "loss"               : "cce",
    "metric"             : "acc",
    
    # Prepare -----------------------------------------------------------------
    
    "patch_size"         : 160,
    "patch_overlap"      : 0,
    "mask_method"        : "edt", # (sm)

    # Train -------------------------------------------------------------------
    
    "display"            : 0,
    "epochs"             : 512,
    "batch_size"         : 32,
    "validation_split"   : 0.2,
    "learning_rate"      : 0.0001,
    "patience"           : 128,

    # Augment -----------------------------------------------------------------
    
    "augment_iterations" : 1024,
    "augment_gamma_p"    : 0.0,
    "augment_gblur_p"    : 0.0,
    "augment_noise_p"    : 0.0,
    "augment_flip_p"     : 0.5,
    "augment_distort_p"  : 0.5,
    
    "augment_parameters" : {
        
    # Gamma
    "gamma_low"          : 0.75,
    "gamma_high"         : 1.25,
    "gamma_chn"          : "independent",
    
    # Gaussian blur
    "gblur_sigma_low"    : 1,
    "gblur_sigma_high"   : 2,
    "gblur_chn"          : "shared",
    
    # Noise
    "noise_gain_low"     : 30,
    "noise_gain_high"    : 60,
    "noise_std_low"      : 3,
    "noise_std_high"     : 6,
    "noise_chn"          : "independent",
    
    # Grid distort
    "distort_steps_low"  : 1,
    "distort_steps_high" : 10,
    "distort_limit_low"  : 0.1,
    "distort_limit_high" : 0.5,
        
    },

    }

#%% Comments ------------------------------------------------------------------

"""
- Make tensors for predictions as well?
- Make a None parameter for patches (patching at the size of input)

"""
    
#%% Class(Model) ---------------------------------------------------------------

class Model:
    
    def __init__(self, parameters=None, model_path=None):
        self.parameters = parameters
        self.model_path = model_path
        
        # Run
        self.initialize()

#%% Class(Model) function(s) ---------------------------------------------------

    def get_parameters(self):
        for key, val in self.parameters.items():
            setattr(self, key, val)
        
    def get_model_path(self):
        if self.root_path is None:
            self.root_path = Path.cwd()
        self.model_path = self.root_path / self.model_name
        
    def load_model_weights(self):
        self.weights_path = self.model_path / "weights.keras"
        if self.weights_path.exists():
            print(f"({self.model_path.name}) : load weights ")
            self.model.load_weights(self.weights_path)
        
#%% Class(Model) initialize() --------------------------------------------------

    def initialize(self):
        
        # Check inputs
        if (self.parameters is None) == (self.model_path is None):
            raise ValueError(
                "User must either provide 'parameters' or 'model_path', but not both"
                )
                    
        # Load parameters (if not provided)
        if self.parameters is None:
            
            with open(self.model_path / "parameters.pkl", "rb") as file:
                self.parameters = pickle.load(file)
                print(f"({self.model_path.name}) : load parameters ")

        # Set input shape
        patch_size = self.parameters["patch_size"]
        n_channels = self.parameters["input_shape"][-1]
        self.parameters["input_shape"] = (
            patch_size, patch_size, n_channels)

        self.get_parameters()
        
    def initialize_train(self):
                            
        if self.model_path is None:

            # Default model name (if not provided)
            if self.model_name is None:
                
                n = self.X.shape[0]
                n_trn = int(n - (n * self.validation_split))
                
                if self.model_type == "sm":
                    var_str = (
                        f"{self.mask_method}_"
                        )
                    
                if self.model_type == "cls":
                    var_str = (
                        f"{len(self.filters)}_"
                        f"{self.n_classes}_"
                        )
                
                if self.model_type == "aec":
                    var_str = (
                        f"{len(self.filters)}_"
                        f"{self.latent_size}_"
                        )
                                
                self.model_name = (
                    f"model-{self.model_type}_"
                    f"{self.patch_size}_"
                    f"{var_str}"
                    f"{n_trn}-"
                    f"{self.augment_iterations}"
                    )
                    
                self.get_model_path()
                
            else:
                
                self.get_model_path()
                self.load_model_weights()
                
        else:
            
            self.load_model_weights()
        
        # Create model directory
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True)

        # Save parameters
        with open(self.model_path / "parameters.pkl", "wb") as file:
            pickle.dump(self.parameters, file)
                    
#%% Class(Model) train() -------------------------------------------------------

    def train(self, X, y=None):
        
        # Check
        self.X, self.y = X, y
        Check(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )
        
        # Prepare
        Prepare(self, self.X, y=self.y, display=self.display)
        
        # Build
        Build(self, model_type=self.model_type)
        
        # Initialize training
        self.initialize_train()

        # Callbacks
        self.callbacks = [CallBacks(self)]
        
        try:
                   
            # Train
            self.history = self.model.fit(
                self.trn_tensor,
                validation_data=self.val_tensor,
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
                
#%% Class(Model) predict() -----------------------------------------------------

    def predict(
            self, X, 
            patch_overlap=None, 
            batch_size=32, 
            chunk_size=None, 
            latent=False
            ):
        
        # Build
        Build(self, model_type=self.model_type)
        self.load_model_weights()
        
        # Select model
        if latent and self.model_type == "aec":
            model = self.model_enc
        else:
            model = self.model
        
        # Initialize
        
        if patch_overlap is None:
            if self.model_type in ["sm", "aec"]:
                patch_overlap = self.patch_size // 2
            elif self.model_type == "cls":
                patch_overlap = 0
        
        if self.input_shape[-1] > 1:
            multichannel_in = True
            if self.model_type == "sm":
                multichannel_out = False
            if self.model_type == "aec":
                multichannel_out = True
        else:
            multichannel_in = False
            multichannel_out = False
        
        Check(X, name="X", ctype=(np.ndarray, list), dtype=float, vrange=(0, 1))
        
        # Convert X to list
        if isinstance(X, list):
            islist = True
        else:
            islist = False
            X = [X]
            
        prds = []
        for arr in X:
            
            if self.model_type == "sm" and multichannel_in:
                shape = arr.shape[:-1] 
            else:
                shape = arr.shape
            
            # Extract patches
            patches = np.stack(extract_patches(
                arr, self.patch_size, patch_overlap, 
                multichannel=multichannel_in
                ))
        
            print(patches.shape)
        
            # Predict
            
            if chunk_size:
                
                n_chunks = int(np.ceil(patches.shape[0] / chunk_size))
                print(f"n_chunks = {n_chunks}")
                
                prd = []
                for i, idx in enumerate(range(0, len(patches), chunk_size)):
                    chunk = patches[idx:idx + chunk_size]
                    prd.append(model.predict(chunk, batch_size=batch_size))
                prd = np.concatenate(prd, axis=0)
                
                del chunk
            
            else:
                
                prd = model.predict(patches, batch_size=batch_size)
                
                del patches
                
            # Append predictions
            
            if self.model_type == "cls":
                prds.append(prd)
            
            elif latent and self.model_type == "aec":
                prds.append(prd)
            
            else:
                
                # Merge patches
                prd = merge_patches(
                    prd, shape, patch_overlap, multichannel=multichannel_out)
                prds.append(prd)
    
        return prds if islist else prds[0]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    from bdtools.test import load_data
    from skimage.transform import downscale_local_mean, resize
    
    # Paths
    # dataset = "em_mito"
    # dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    # dataset = "sat_roads"
    dataset = "chess_class"
    
    # Load data
    X, y = load_data(dataset)
    
    # Downscale (optional)
    if dataset == "chess_class":
        X = downscale_local_mean(X, (1, 4, 4, 1))
    
#%% train() -------------------------------------------------------------------
    
    model = Model(parameters=parameters, model_path=None)
    model.train(X, y=y)       

#%% predict() -----------------------------------------------------------------
        
    # model_path = Path(Path.cwd(), "model-cls_160_2_12_240-1024")
    # model = Model(parameters=None, model_path=model_path)
    # prds = model.predict(
    #     X, patch_overlap=None, batch_size=32, chunk_size=None, latent=False)
    
    # if dataset == "chess_class":
    #     prds = resize(prds, (prds.shape[0] / 10, prds.shape[1] * 10))
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # if isinstance(X, list):
    #     idx = 0
    #     # vwr.add_image(X[idx])
    #     vwr.add_image(prds[idx])
    # else:
    #     # vwr.add_image(X)
    #     vwr.add_image(prds)
        
#%%

    # import matplotlib.pyplot as plt
    # from tensorflow.keras.utils import to_categorical
    # from sklearn.metrics import (
    #     confusion_matrix, classification_report, accuracy_score)

    # # Classes
    # classes = [
    #     "bis_b", "bis_w",
    #     "kin_b", "kin_w",
    #     "kni_b", "knt_w",
    #     "paw_b", "paw_w",
    #     "que_b", "que_w",
    #     "roo_b", "roo_w",
    #     ]

    # # One hot encoding    
    # y_hot = to_categorical(y, num_classes=parameters["n_classes"])

    # # Get classes
    # y_pred = np.argmax(prds, axis=1)
    # y_true = np.argmax(y_hot, axis=1)
    
    # # Confusion matrix & report
    # cmat = confusion_matrix(y_true, y_pred).astype("float32")
    # cmat /= cmat.sum(axis=1, keepdims=True)
    # stat = classification_report(y_true, y_pred, output_dict=True)
    
    # # Fetch
    # stat_str = []
    # stat_str.append(" cls    prc     rec     f1s  ")
    # stat_str.append("----------------------------")
    # for c in range(model.n_classes):
    #     if classes is not None:
    #         cls_str = f"({ classes[c]})"
    #     else:
    #         cls_str = f"()"
    #     stat_str.append(
    #         f" {c:03d} | "
    #         f"{stat[str(c)]['precision']:.3f} | "
    #         f"{stat[str(c)]['recall'   ]:.3f} | "
    #         f"{stat[str(c)]['f1-score' ]:.3f} "
    #         f"{cls_str}"
    #         )
    # stat_str.append("----------------------------")
    # stat_str.append(
    #     " avg | "
    #     f"{stat['macro avg']['precision']:.3f} | "
    #     f"{stat['macro avg']['recall'   ]:.3f} | "
    #     f"{stat['macro avg']['f1-score' ]:.3f}"
    #     )
    # stat_str.append(
    #     "wavg | "
    #     f"{stat['weighted avg']['precision']:.3f} | "
    #     f"{stat['weighted avg']['recall'   ]:.3f} | "
    #     f"{stat['weighted avg']['f1-score' ]:.3f}"
    #     )
    # stat_str = "\n".join(stat_str)
    # print(stat_str)
        
    # # Plot --------------------------------------------------------------------
    
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6)) 
    # img = ax.imshow(cmat, interpolation="nearest")
    # ax.figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    
    # ax.text(
    #     0.0, -0.25, stat_str, 
    #     size=12, color="k", font="consolas",
    #     transform=ax.transAxes, ha="left", va="top",
    #     )

    # ax.set_title(model.model_name)
    # ax.set_xticks(np.arange(cmat.shape[1]))
    # ax.set_yticks(np.arange(cmat.shape[0]))
    # ax.set_xticklabels(classes, rotation=90)
    # ax.set_yticklabels(classes)
    # ax.set_xlabel("True")
    # ax.set_ylabel("Predicted")

