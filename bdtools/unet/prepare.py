#%% Imports -------------------------------------------------------------------

#%% Class(Prepare) ------------------------------------------------------------

class Prepare:
    
    def __init__(self, unet):
        self.unet = unet
                
        # Run
        self.initialize()
        self.normalize()

    def __getattr__(self, name):
        return getattr(self.unet, name)

#%% Class(Prepare) initialize() -----------------------------------------------

    def initialize(self):
        pass
    
#%% Class(Prepare) normalize() ------------------------------------------------

    def normalize(self):
        
        if self.normalize == "none":
            print("none")
        
        pass
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pass