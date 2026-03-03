#%% Imports -------------------------------------------------------------------

import numpy as np

#%% Class(Check_parameter) ----------------------------------------------------

class Check_parameter:
    
    """
    A utility class to validate parameters against specific types, values, 
    ranges, and dimensions. Used for early-exit error handling.

    Parameters
    ----------
    value : any
        The parameter value to be checked.
    name : str
        The name of the parameter (used in error messages).
    ctype : class type or tuple of class types, optional
        The expected class type(s) of the 'value' (e.g., int, str, np.ndarray).
    dtype : class type or tuple of class types, optional
        The expected numerical data type(s) if 'value' is a NumPy array 
        or a list of arrays. Restricted to int, float, and/or bool.
    vvalue : iterable, optional
        A collection of allowed specific values (e.g., a list of valid strings).
    vrange : tuple, optional
        A tuple (min, max) defining the inclusive range for numerical values.
    ndim : int or tuple of ints, optional
        The allowed number of dimensions if 'value' is a NumPy array or 
        a list of arrays.
        
    Raises
    ------
    TypeError
        If ctype, dtype, or ndim requirements are not met.
    ValueError
        If vvalue or vrange requirements are not met.
    
    """
    
    def __init__(
        self, value, name="", 
        ctype=None, dtype=None,
        vvalue=None, vrange=None, ndim=None,
        ):
        
        # Fetch
        self.value  = value
        self.name   = name
        self.ctype  = ctype
        self.dtype  = dtype
        self.vvalue = vvalue
        self.vrange = vrange
        self.ndim   = ndim
                
        # Run
        self.initialize()
        if ctype  is not None: self.check_ctype()
        if dtype  is not None: self.check_dtype()
        if vvalue is not None: self.check_vvalue()
        if vrange is not None: self.check_vrange()
        if ndim   is not None: self.check_ndim()
                               
    #     if vrange is not None:
    # # If it's a collection, check if it has items. 
    # # If it's just a number, it's automatically "not empty".
    # if not hasattr(vrange, "__len__") or len(vrange) > 0:
    #     self.check_vrange()
        
#%% Class(Check_parameter) : initialize ---------------------------------------
            
    def initialize(self):
        
        # Format inputs
        if not isinstance(self.ctype, tuple):
            self.ctype = tuple([self.ctype])
        if not isinstance(self.dtype, tuple):
            self.dtype = tuple([self.dtype])
        if not isinstance(self.vvalue, tuple): 
            self.vvalue = tuple([self.vvalue])
        if not isinstance(self.vrange, tuple): 
            self.vrange = tuple([self.vrange])
        if not isinstance(self.ndim, tuple): 
            self.ndim = tuple([self.ndim])

        # Expected dtypes
        if self.dtype is not None:
            dtypes = []
            if int in self.dtype:
                dtypes.extend([
                    np.int64, np.int32, np.int16, np.int8,
                    np.uint64, np.uint32, np.uint16, np.uint8,
                    ])
            if float in self.dtype:
                dtypes.extend([
                    np.float64, np.float32, np.float16
                    ])
            if bool in self.dtype:
                dtypes.append(np.bool_)
            self.dtypes = tuple(dtypes)

#%% Class(Check_parameter) : methods ------------------------------------------
            
    def check_ctype(self):
        ctype = type(self.value)       
        if self.ctype is not None:
            if not isinstance(self.value, self.ctype):
                raise TypeError(
                    f"Invalid '{self.name}' ctype {ctype!r}, "
                    f"expected {self.ctype!r}"
                    )
                
    def check_dtype(self):
        ctype = type(self.value) 
        if ctype == np.ndarray:
            dtype = self.value.dtype.type
            if dtype not in self.dtypes:
                raise TypeError(
                    f"Invalid '{self.name}' dtype {dtype!r}, "
                    f"expected {self.dtypes!r}"
                    )
        if ctype == list:
            for i, arr in enumerate(self.value):
                dtype = arr.dtype.type
                if dtype not in self.dtypes:
                    raise TypeError(
                        f"Invalid '{self.name}' dtype {dtype!r} at index [{i}], "
                        f"expected {self.dtypes!r}"
                        )
                
    def check_vvalue(self):
        if not self.value in self.vvalue:
            raise ValueError(
                f"Invalid '{self.name}' value {self.value!r}, "
                f"expected {self.vvalue!r}"
                )
            
    def check_vrange(self, atol=1e-5):
        val = np.asanyarray(self.value)
        v0 = self.vrange[0] - atol
        v1 = self.vrange[1] + atol
        if np.any((val < v0) | (val > v1)):
            raise ValueError(
                f"{self.name!r} out of range" 
                f"({np.min(self.value):.3f}, {np.max(self.value):.3f}), "
                f"expected range {self.vrange!r}"
                )
            
    def check_ndim(self):
        ctype = type(self.value) 
        if ctype == np.ndarray:
            ndim = self.value.ndim
            if ndim not in self.ndim:
                raise TypeError(
                    f"Invalid '{self.name}' ndim ({ndim!r}), "
                    f"expected {self.ndim!r}"
                    )
        if ctype == list:
            for i, arr in enumerate(self.value):
                ndim = arr.ndim
                if ndim not in self.ndim:
                    raise TypeError(
                        f"Invalid '{self.name}' ndim ({ndim!r}) at index [{i}], "
                        f"expected {self.ndim!r}"
                        )
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    value_0 = "binary"
    Check_parameter(
        value_0, name="mask_type", ctype=str,
        vvalue=("binary", "edt", "outlines"),
        )    
    
    value_1 = 5
    Check_parameter(
        value_1, name="thresh", ctype=int,
        vrange=(0, 10),
        )
    
    value_2 = np.full((128, 128, 128), 2)
    Check_parameter(
        value_2, name="input_2", dtype=int,
        ndim=(2, 3), vrange=(0, 1)
        )
    
    # value_3 = []
    # for _ in range(10):
    #     value_3.append(np.full((128, 128, 128), 1))
    # Check_parameter(
    #     value_3, name="input_3", dtype=int,
    #     ndim=(1, 2),
    #     )
    