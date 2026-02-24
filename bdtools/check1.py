#%% Imports -------------------------------------------------------------------

import numpy as np

#%% Class: Check_parameter() --------------------------------------------------

class Check_parameter:
    
    def __init__(
        self, value, name=None, ctype=None, valid=None, vrange=None):
        
        # Fetch
        self.value  = value
        self.name   = name
        self.ctype  = ctype
        self.valid  = valid
        self.vrange = vrange
        
        # Run
        self.check_type()
        if self.valid is not None:
            self.check_valid()
        if self.vrange is not None:
            self.check_vrange()
                    
    def check_type(self):
        ctype = type(self.value)       
        if self.ctype is not None:
            if not isinstance(self.value, self.ctype):
                raise TypeError(
                    f"Invalid '{self.name}' {ctype!r}, "
                    f"expected {self.ctype!r}"
                    )
                
    def check_valid(self):
        if not self.value in self.valid:
            raise ValueError(
                f"Invalid '{self.name}' {self.value!r}, "
                f"expected {self.valid!r}"
                )
            
    def check_vrange(self):
        if not (self.vrange[0] <= self.value <= self.vrange[1]):
            raise ValueError(
                f"Invalid {self.name!r} {self.value!r}, "
                f"expected range {self.vrange!r}"
                )

#%% Class: Check_data() -------------------------------------------------------

class Check_data:
    
    def __init__(
        self, value, name=None, ctype=None, dtype=None, ndim=None):
        
        # Fetch
        self.value = value
        self.name  = name
        self.ctype = ctype
        self.dtype = dtype
        self.ndim  = ndim
        
        if not isinstance(self.dtype, list):
            self.dtype = [self.dtype]
        
        # Expected dtypes
        self.dtypes = []
        if int in self.dtype:
            self.dtypes.extend([
                np.int64, np.int32, np.int16, np.int8,
                np.uint64, np.uint32, np.uint16, np.uint8,
                ])
        elif float in self.dtype:
            self.dtypes.extend([
                np.float64, np.float32, np.float16
                ])
        elif bool in self.dtype:
            self.dtypes.extend([
                np.bool_
                ])
        
        # Run
        self.check_type()
        if self.ndim is not None:
            self.check_ndim()
        
    def check_type(self):

        ctype = type(self.value) 
        
        if not isinstance(self.value, (np.ndarray, list)):
            raise TypeError(
                f"Invalid '{self.name}' {ctype!r}, "
                f"expected {(np.ndarray, list)!r}"
                )
        
        if self.dtype is not None:
            
            if ctype == np.ndarray:
                dtype = self.value.dtype.type
                if dtype not in self.dtypes:
                    raise TypeError(
                        f"Invalid '{self.name}' {dtype!r}, "
                        f"expected {self.dtypes!r}"
                        )
                    
            if ctype == list:
                for i, arr in enumerate(self.value):
                    dtype = arr.dtype.type
                    if dtype not in self.dtypes:
                        raise TypeError(
                            f"Invalid '{self.name}' {dtype!r} at index [{i}], "
                            f"expected {self.dtypes!r}"
                            )
                        
    def check_ndim(self):
        
        ctype = type(self.value) 
        
        if ctype == np.ndarray:
            ndim = self.value.ndim
            if ndim not in self.ndim:
                raise TypeError(
                    f"Invalid '{self.name}' ndim [{ndim!r}], "
                    f"expected {self.ndim!r}"
                    )
                
        if ctype == list:
            for i, arr in enumerate(self.value):
                ndim = arr.ndim
                if ndim not in self.ndim:
                    raise TypeError(
                        f"Invalid '{self.name}' ndim [{ndim!r}] at index [{i}], "
                        f"expected {self.ndim!r}"
                        )
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    value_0 = "binary"
    Check_parameter(
        value_0, name="mask_type", ctype=str,
        valid=["binary", "edt", "outlines"],
        )    
    
    value_1 = 5
    Check_parameter(
        value_1, name="thresh", ctype=int,
        vrange=[0, 10],
        )
    
    value_2 = np.full((128, 128, 128), 1)
    Check_data(
        value_2, name="input_2", dtype=int,
        ndim=[1, 2],
        )
    
    value_3 = []
    for _ in range(10):
        value_3.append(np.full((128, 128, 128), 1))
    Check_data(
        value_3, name="input_3", dtype=int,
        ndim=[1, 2],
        )
    