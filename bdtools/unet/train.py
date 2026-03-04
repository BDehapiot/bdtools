#%% Imports -------------------------------------------------------------------

from bdtools.unet.build import Build

#%% Class(Train) --------------------------------------------------------------

class Train:
    
    def __init__(self, unet):
        self.unet = unet
        self.parameters = unet.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self.model = Build(self).model
        self._train()
        
#%% Class(Train) function(s) --------------------------------------------------

    def _train(self):

        self.history = self.model.fit(
            x=self.unet.X_trn, y=self.unet.y_trn,
            validation_data=(self.unet.X_val, self.unet.y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=0,
            )        