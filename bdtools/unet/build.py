#%% Imports -------------------------------------------------------------------

import segmentation_models as sm

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
        
        # Finalize
        self.unet.model = self.model
        
#%% Class(Build) function(s) --------------------------------------------------

    def _build(self):

        # Build
        self.model = sm.Unet(
            self.build_params["backbone"], 
            input_shape=(None, None, 1), # Parameter
            classes=self.build_params["classes"],
            activation=self.build_params["activation"],
            encoder_weights=None,
            )
