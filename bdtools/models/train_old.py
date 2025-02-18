#%% Imports -------------------------------------------------------------------

import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#%% Function : split() --------------------------------------------------------

def split(X, y, val_split=0.2):
    n_total = X.shape[0]
    n_val = int(n_total * val_split)
    idx = np.random.permutation(np.arange(0, n_total))
    X_trn, y_trn = X[idx[n_val:]], y[idx[n_val:]]
    X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
    return X_trn, y_trn, X_val, y_val

#%% Function : build() --------------------------------------------------------

def build_unet(
        input_shape=(512, 512, 1),
        num_classes=1,
        filters=[16, 32, 64, 128],
        kernel_size=(3, 3),
        pool_size=(2, 2),
        learning_rate=0.001
        ):
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoding path
    convs = []
    x = inputs
    for f in filters:
        x = layers.Conv2D(f, kernel_size, activation='relu', padding='same')(x)
        x = layers.Conv2D(f, kernel_size, activation='relu', padding='same')(x)
        # x = layers.Dropout(0.3)(x)
        # x = layers.BatchNormalization()(x)
        convs.append(x)
        x = layers.MaxPooling2D(pool_size=pool_size)(x)
    
    # Bottleneck
    x = layers.Conv2D(
        filters[-1] * 2, kernel_size, activation='relu', padding='same')(x)
    x = layers.Conv2D(
        filters[-1] * 2, kernel_size, activation='relu', padding='same')(x)
    
    # Decoding path
    for f, conv in zip(reversed(filters), reversed(convs)):
        x = layers.Conv2DTranspose(
            f, pool_size, strides=pool_size, padding='same')(x)
        x = layers.concatenate([x, conv])  # Skip connection
        x = layers.Conv2D(
            f, kernel_size, activation='relu', padding='same')(x)
        x = layers.Conv2D(
            f, kernel_size, activation='relu', padding='same')(x)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
            
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
        )
            
    return model

#%% Function : train() --------------------------------------------------------

def train_unet(
        X_trn, y_trn, 
        X_val, y_val,
        build_params,
        epochs=100,
        batch_size=32,
        patience=20,     
        ):
            
    # Add channel dimension    
    X_trn = np.expand_dims(X_trn, axis=-1)  
    X_val = np.expand_dims(X_val, axis=-1)  
    y_trn = np.expand_dims(y_trn, axis=-1)  
    y_val = np.expand_dims(y_val, axis=-1)  
    
    # Build U-Net model
    model = build_unet(**build_params)
    model.summary()
        
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True)
    # model_checkpoint = ModelCheckpoint(
    #     best_path, monitor='val_loss',
    #     save_best_only=True, mode='min',
    #     )
    
    # Train U-Net
    history = model.fit(
        X_trn, y_trn, 
        validation_data=(X_val, y_val),
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=[early_stopping]
        )
       
    # Save weights
    # model.save_weights(model_path)
    
    return history

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path
    from bdtools.models import preprocess

    # Parameters
    dataset = "em_mito"
    # dataset = "fluo_nuclei"
    patch_size = 256
    patch_overlap = 0
    
    # Build parameters
    build_params = {
        "input_shape" : (patch_size, patch_size, 1),
        "num_classes" : 1,
        "filters" : [64, 128, 256, 512],
        "kernel_size" : (3, 3),
        "pool_size" : (2, 2),
        "learning_rate" : 0.001
        }
    
    # Train parameters
    train_params = {
        "epochs" : 100,
        "batch_size" : 8,
        "patience" : 20, 
        }
    
    # Paths
    local_path = Path.cwd().parent.parent / "_local"
    img_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
    msk_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    
    # Load images & masks
    X = io.imread(img_path)
    y = io.imread(msk_path)
    
    # Preprocess
    t0 = time.time()
    print("preprocess :", end=" ", flush=True)
    X, y = preprocess(
        X, msks=y, 
        img_norm="global",
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Split
    X_trn, y_trn, X_val, y_val = split(X, y, val_split=0.2)
    
    # Train
    history = train_unet(
        X_trn, y_trn, 
        X_val, y_val,
        build_params,
        **train_params
        )
    
    pass