#%% Imports -------------------------------------------------------------------

#%% Class: Check_parameter() --------------------------------------------------

class Check_parameter:
    
    def __init__(
        self, value, name=None, 
        ctype=None, dtype=None, 
        valid=None, vrange=None,
        ):
        
        # Fetch
        self.value = value
        self.name  = name
        self.ctype = ctype
        self.valid = valid
        self.vrange= vrange
        
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
                    f"Invalid {self.name!r} type {ctype!r}, "
                    f"expected {self.ctype!r}"
                    )
                
    def check_valid(self):
        if not self.value in self.valid:
            raise ValueError(
                f"Invalid {self.name!r} value {self.value!r}, "
                f"expected {self.valid!r}"
                )
            
    def check_vrange(self):
        if not (self.vrange[0] <= self.value <= self.vrange[1]):
            raise ValueError(
                f"Invalid {self.name!r} value {self.value!r}, "
                f"expected range {self.vrange!r}"
                )

#%% Class: Check_data() -------------------------------------------------------

class Check_data:
    
    def __init__(
        self, value, name=None, dtype=None, 
        ndim=None, 
        ):
        
        # Fetch
        self.value = value
        self.name  = name
        self.dtype = dtype
        self.ndim  = ndim
        
        # Run
        self.check_type()
        # if self.ndim is not None:
        #     self.check_ndim()
        

    def check_type(self):
        ctype = type(self.value) 
        if not isinstance(self.value, (np.ndarray, list)):
            raise TypeError(
                f"Invalid {self.name!r} type {ctype!r}, "
                f"expected {(np.ndarray, list)!r}"
                )
        # if self.dtype is not None:
        #     if ctype == np.ndarray:
        #         dtype = self.value.dtype
        #         if dtype not in self.dtype:
        #             raise TypeError(
        #                 f"Invalid {self.name!r} data type '{dtype.name}', "
        #                 f"expected {self.dtype!r}"
        #                 )
            
        # # Encoding
        # if type(self.value) == list:
        #     for i, arr in enumerate(self.value):
        #         if not arr.dtype in self.dtype:
        #             raise TypeError(
        #                 "\n"
        #                 f"parameter '{self.name}' encoding type at list idx[{i}] is {arr.ndim}"
        #                 "\n"
        #                 f"expected ndim {self.ndim}"
        #                 )
                    
    # def check_ndim(self):
        
    #     if type(self.value) == list:
    #         for i, arr in enumerate(self.value):
    #             if not arr.ndim in self.ndim:
    #                 raise TypeError(
    #                     "\n"
    #                     f"parameter '{self.name}' ndim at list idx[{i}] is {arr.ndim}"
    #                     "\n"
    #                     f"expected ndim {self.ndim}"
    #                     )
        
    #     elif type(self.value) == np.ndarray:
    #         if not self.value.ndim in self.ndim:
    #             raise TypeError(
    #                 "\n"
    #                 f"parameter '{self.name}' ndim is {self.value.ndim}"
    #                 "\n"
    #                 f"expected ndim {self.ndim}"
    #                 )
                    
            # if not self.value.ndim in self.ndim:
            #     raise TypeError(
            #         "\n"
            #         f"parameter '{self.name}' ndim is {self.value.ndim}"
            #         "\n"
            #         f"expected ndim {self.ndim}"
            #         )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import numpy as np
    
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
    
    value_2 = np.full((128, 128, 128), 0.5)
    value_2 = 1
    Check_data(
        value_2, name="arr", dtype=np.integer,
        ndim=[2, 3],
        )
    
    # value_3 = [np.full((128, 128, 128), 0.5) for _ in range(10)]
    # print(value_3.dtype)
    # Check_data(
    #     value_3, name="arr", dtype=np.floating,
    #     ndim=(2, 3),
    #     )
    