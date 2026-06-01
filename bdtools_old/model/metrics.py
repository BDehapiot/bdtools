#%% Imports -------------------------------------------------------------------

from tensorflow.keras import backend as K

#%% Function(s): loss ---------------------------------------------------------

def mse(y_true, y_pred):
    
    """
    Mean Squared Error (MSE)
    ------------------------
    
    Measures the average of the squares of the errors.
    Useful for penalizing larger outliers in reconstruction.
    y_true: ground truth image/mask.
    y_pred: predicted image/mask.
    
    """
    
    return K.mean(K.square(y_pred - y_true))

def mae(y_true, y_pred):
    
    """
    Mean Absolute Error (MAE)
    -------------------------
    
    Measures the average magnitude of the errors.
    More robust to outliers than MSE.
    y_true: ground truth image/mask.
    y_pred: predicted image/mask.
    
    """
    
    return K.mean(K.abs(y_pred - y_true))

def bce(y_true, y_pred):
    
    """
    Binary Cross-Entropy (BCE)
    --------------------------
    
    Measures the distance between true labels and predicted probabilities.
    Effective for forcing the model to distinguish between foreground and background.
    y_true: ground truth image/mask (scaled 0-1).
    y_pred: predicted image/mask (scaled 0-1).
    
    """
    
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def cce(y_true, y_pred):
    
    """
    Categorical Cross-Entropy (CCE)
    -------------------------
    
    Measures the cross-entropy loss between true labels (one-hot) 
    and predicted probabilities.
    
    """
    return K.mean(K.categorical_crossentropy(y_true, y_pred))

def scce(y_true, y_pred):
    
    """
    Sparse Categorical Cross-Entropy (SCCE)
    --------------------------------
    
    Measures loss when labels are provided as integers.
    """
    return K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))

#%% Function(s): accuracy -----------------------------------------------------

from tensorflow.keras.metrics import (
    categorical_accuracy, sparse_categorical_accuracy)

def acc(y_true, y_pred):
    
    """
    Accuracy (ACC)
    --------
    
    Calculates the mean accuracy rate across all predictions.
    Assumes one-hot encoded labels.
    
    """
    return K.mean(categorical_accuracy(y_true, y_pred))

def sacc(y_true, y_pred):
    
    """
    Sparse Accuracy (SACC)
    ---------------
    
    Calculates the mean accuracy rate for integer-encoded labels.
    
    """
    return K.mean(sparse_categorical_accuracy(y_true, y_pred))

#%% Function(s): segmentation -------------------------------------------------

def hdc(y_true, y_pred, threshold=0.5, smooth=1e-6):
    
    """
    Hard Dice Coefficient (HDC)
    ---------------------
    
    Measures overlap after thresholding inputs.
    y_true: binary/probability mask.
    y_pred: probability mask.
    
    """
    
    y_pred_bin = K.cast(y_pred > threshold, "float32")
    y_true_bin = K.cast(y_true > threshold, "float32")
    intersection = K.sum(y_true_bin * y_pred_bin)
    
    return (2. * intersection + smooth) / (K.sum(y_true_bin) + K.sum(y_pred_bin) + smooth)

def sdc(y_true, y_pred, smooth=1e-6):
    
    """
    Soft Dice Coefficient (SDE)
    ---------------------
    
    Measures overlap on probability maps without thresholding.
    y_true: binary/probability mask.
    y_pred: probability mask.
    
    """
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    
    """
    Intersection Over Union (IoU)
    -----------------------------
    
    Measures the overlap vs. union of masks.
    y_true: binary/probability mask.
    y_pred: probability mask.
    
    """
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision(y_true, y_pred, threshold=0.5):
    
    """
    Precision
    ---------
    
    Ratio of true positives over predicted positives.
    Use when false positives are costly.
    y_true: binary/probability mask.
    y_pred: probability mask.
    
    """
    
    y_pred_bin = K.cast(y_pred > threshold, 'float32')
    y_true_bin = K.cast(y_true > threshold, 'float32')
    true_pos = K.sum(y_true_bin * y_pred_bin)
    pred_pos = K.sum(y_pred_bin)
    return true_pos / (pred_pos + K.epsilon())

def recall(y_true, y_pred, threshold=0.5):
    
    """
    Recall
    ------
    
    Ratio of true positives over actual positives.
    Use when false negatives are costly.
    y_true: binary/probability mask.
    y_pred: probability mask.
    
    """
    
    y_pred_bin = K.cast(y_pred > threshold, 'float32')
    y_true_bin = K.cast(y_true > threshold, 'float32')
    true_pos = K.sum(y_true_bin * y_pred_bin)
    actual_pos = K.sum(y_true_bin)
    return true_pos / (actual_pos + K.epsilon())

def f1(y_true, y_pred, threshold=0.5):
    
    """
    F1 score
    --------
    
    Harmonic mean of precision and recall.
    y_true: binary/probability mask.
    y_pred: probability mask.
    
    """
    
    prec = precision(y_true, y_pred, threshold)
    rec = recall(y_true, y_pred, threshold)
    return 2 * (prec * rec) / (prec + rec + K.epsilon())
