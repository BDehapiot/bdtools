#%% Imports -------------------------------------------------------------------

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

import segmentation_models as sm

# bdtools 
import metrics

# Tensorflow
from tensorflow.keras.optimizers import Adam

#%% Class(Build) --------------------------------------------------------------

class Build:
    
    def __init__(self, unet):
        self.unet = unet
        self.parameters = unet.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self._build()
        
#%% Class(Build) function(s) --------------------------------------------------

    def _build(self):

        # Build
        model = sm.Unet(
            self.backbone, 
            input_shape=(None, None, 1), # Parameter
            classes=1, # Parameter
            activation=self.activation,
            encoder_weights=None,
            )
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy", # Parameter
            metrics=[getattr(metrics, self.metric)],
            )
        
        return model
