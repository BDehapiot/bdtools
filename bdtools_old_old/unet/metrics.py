#%% Imports -------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import backend as K

#%% Function(s): metrics -----------------------------------------------------

def dice_coef(y_true, y_pred, threshold=0.5, smooth=1e-6):
    
    """
    Hard Dice coef for segmentation.
    
    Measures overlap after thresholding inputs.
    y_true: binary/probability mask.
    y_pred: probability mask.
    """
    
    y_pred_bin = K.cast(y_pred > threshold, 'float32')
    y_true_bin = K.cast(y_true > threshold, 'float32')
    intersection = K.sum(y_true_bin * y_pred_bin)
    return (2. * intersection + smooth) / (K.sum(y_true_bin) + K.sum(y_pred_bin) + smooth)

def soft_dice_coef(y_true, y_pred, smooth=1e-6):
    
    """
    Soft Dice coef for segmentation.
    
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
    Intersection over Union (IoU) metric.
    
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
    Precision metric.
    
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
    Recall metric.
    
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

def f1_score(y_true, y_pred, threshold=0.5):
    
    """
    F1 score metric.
    
    Harmonic mean of precision and recall.
    y_true: binary/probability mask.
    y_pred: probability mask.
    """
    
    prec = precision(y_true, y_pred, threshold)
    rec = recall(y_true, y_pred, threshold)
    return 2 * (prec * rec) / (prec + rec + K.epsilon())

def brier_score(y_true, y_pred):
    
    """
    Brier score.
    
    Mean squared error between predicted probabilities and labels.
    Lower score indicates better calibration.
    y_true: binary mask.
    y_pred: probability mask.
    """
    
    return tf.reduce_mean(tf.square(y_pred - y_true))
