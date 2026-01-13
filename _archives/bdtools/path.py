#%% Imports -------------------------------------------------------------------

#%% Function: get_paths() -----------------------------------------------------

def get_paths(
        rootpath, 
        ext=".tif", 
        tags_in=[], 
        tags_out=[], 
        subfolders=False, 
        ):
    
    """     
    Retrieve file paths with specific extensions and tag criteria from a 
    directory. The search can include subfolders if specified.
    
    Parameters
    ----------
    rootpath : str or pathlib.Path
        Path to the target directory where files are located.
        
    ext : str, default=".tif"
        File extension to filter files by (e.g., ".tif" or ".jpg").
        
    tags_in : list of str, optional
        List of tags (substrings) that must be present in the file path
        for it to be included.
        
    tags_out : list of str, optional
        List of tags (substrings) that must not be present in the file path
        for it to be included.
        
    subfolders : bool, default=False
        If True, search will include all subdirectories within `rootpath`. 
        If False, search will be limited to the specified `rootpath` 
        directory only.
        
    Returns
    -------  
    selected_paths : list of pathlib.Path
        A list of file paths that match the specified extension and 
        tag criteria.
        
    """
    
    if subfolders:
        paths = list(rootpath.rglob(f"*{ext}"))
    else:
        paths = list(rootpath.glob(f"*{ext}"))
        
    selected_paths = []
    for path in paths:
        if tags_in: 
            check_tags_in = all(tag in str(path) for tag in tags_in)
        else: 
            check_tags_in = True
        if tags_out: 
            check_tags_out = not any(tag in str(path) for tag in tags_out)
        else: 
            check_tags_out = True
        if check_tags_in and check_tags_out:
            selected_paths.append(path)

    return selected_paths  