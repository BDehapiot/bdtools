#%% Imports -------------------------------------------------------------------

#%% Class: Check_parameter() --------------------------------------------------

class Check_parameter:
    
    def __init__(
        self, value, name=None, dtype=None, 
        valid=None, vrange=None,
        ):
        
        # Fetch
        self.value  = value
        self.name   = name
        self.dtype  = dtype
        self.valid  = valid
        self.vrange = vrange
        
        # Run
        self.check_dtype()
        if self.valid is not None:
            self.check_valid()
        if self.vrange is not None:
            self.check_vrange()
        
    def check_dtype(self):
        if not isinstance(self.value, self.dtype):
            raise TypeError(
                "\n"
                f"parameter '{self.name}' type is '{type(self.value).__name__}'"
                "\n"
                f"expected type '{self.dtype.__name__}'"
                )
            
    def check_valid(self):
        if not self.value in self.valid:
            raise TypeError(
                "\n"
                f"parameter '{self.name}' = '{self.value}'"
                "\n"
                f"expected values {self.valid}"
                )
            
    def check_vrange(self):
        if not (self.vrange[0] <= self.value <= self.vrange[1]):
            raise TypeError(
                "\n"
                f"parameter '{self.name}' = {self.value}"
                "\n"
                f"expected range {self.vrange[0], self.vrange[1]}"
                )

#%% Class: Check_data() -------------------------------------------------------

class Check_parameter:
    
    def __init__(
        self, data, name=None,
        ):
        
        pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    param = "binary"
    Check_parameter(
        param, name="mask_type", dtype=str,
        valid=["binary", "edt", "outlines"],
        )    
    param = 3
    Check_parameter(
        param, name="thresh", dtype=int,
        vrange=[0, 2],
        )