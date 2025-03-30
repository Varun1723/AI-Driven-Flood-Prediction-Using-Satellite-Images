import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ====== CONFIGURATION ======
# Paths
INPUT_DIR = "C:/Varun/Coding/Flood_Prediction/processed_data"  # Where your .npy files are stored
MODEL_DIR = "C:/Varun/Coding/Flood_Prediction/models"          # Where to save models and plots
RESULTS_DIR = "C:/Varun/Coding/Flood_Prediction/results"       # Where to save evaluation results

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 256, 256  # Size of SAR image patches
NUM_CHANNELS = 2                  # VH and VV polarization channels

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# ====== DATA LOADING FUNCTIONS ======
def load_dataset():
    """
    Load the preprocessed SAR images from .npy files and create labels.
    
    Returns:
        X_data (np.array): Array of SAR images with shape (n_samples, height, width, channels)
        y_data (np.array): Binary labels (1 for flood, 0 for non-flood)
    """
    X_data = []
    y_data = []
    
    print("Step 1: Loading dataset from:", INPUT_DIR)
    
    # List all .npy files
    npy_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {INPUT_DIR}")
    
    print(f"Found {len(npy_files)} .npy files")
    
    for file_name in npy_files:
        if file_name == "sar_images.npy":  # Skip the empty file if it exists
            continue
            
        # Load the SAR image (VH, VV channels)
        file_path = os.path.join(INPUT_DIR, file_name)
        sar_image = np.load(file_path)
        
        # Resize if necessary
        if sar_image.shape[0] != IMG_HEIGHT or sar_image.shape[1] != IMG_WIDTH:
            # Use tensorflow to resize
            sar_image = tf.image.resize(sar_image, [IMG_HEIGHT, IMG_WIDTH]).numpy()
        
        # Ensure the array has shape (height, width, channels)
        if len(sar_image.shape) == 2:
            sar_image = np.expand_dims(sar_image, axis=-1)
            
        # Append to dataset
        X_data.append(sar_image)
        
        # Create labels based on filenames
        # IMPORTANT: Adjust this logic based on your actual data labeling
        # Current logic assumes flood-related images have "flood" in the filename
        is_flood = 1 if "flood" in file_name.lower() else 0
        y_data.append(is_flood)
    
    # Convert to numpy arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    # Print dataset statistics
    num_flood = np.sum(y_data)
    num_non_flood = len(y_data) - num_flood
    print(f"Dataset loaded: {len(y_data)} images")
    print(f"  - Flood images: {num_flood} ({num_flood/len(y_data)*100:.1f}%)")
    print(f"  - Non-flood images: {num_non_flood} ({num_non_flood/len(y_data)*100:.1f}%)")
    print(f"  - Image shape: {X_data.shape[1:]} (Height, Width, Channels)")
    
    return X_data, y_data

# ====== MODEL BUILDING FUNCTIONS ======
def build_cnn_model():
    """
    Build and compile the CNN model architecture.
    
    Returns:
        model: Compiled Keras model
    """
    print("Step 2: Building CNN model architecture")
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (flood or no flood)
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Print model summary
    model.summary()
    return model

# ====== TRAINING FUNCTIONS ======
def train_model(X_train, y_train, X_val, y_val):
    """
    Train the CNN model with the provided training and validation data.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        
    Returns:
        model: Trained Keras model
        history: Training history
    """
    print("Step 3: Training CNN model")
    
    # Create model
    model = build_cnn_model()
    
    # Define callbacks for training
    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        
        # Save the best model during training
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print(f"\nStarting training with {EPOCHS} epochs and batch size {BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
    print(f"Model saved to {os.path.join(MODEL_DIR, 'final_model.h5')}")
    
    return model, history

# ====== EVALUATION FUNCTIONS ======
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        
    Returns:
        y_pred: Predicted labels for test data
    """
    print("Step 4: Evaluating model performance")
    
    # Get model predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate and print metrics
    print("\n===== Model Evaluation =====")
    
    # Classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Save classification report to file
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Not Flooded', 'Flooded']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")
    plt.close()
    
    return y_pred

def plot_training_history(history):
    """
    Plot and save training history graphs (accuracy and loss).
    
    Args:
        history: Training history from model.fit()
    """
    print("Step 5: Plotting training history")
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))
    print(f"Training history plots saved to {os.path.join(RESULTS_DIR, 'training_history.png')}")
    plt.close()

def visualize_predictions(X_test, y_test, y_pred, num_samples=5):
    """
    Visualize some correct and incorrect predictions.
    
    Args:
        X_test: Test images
        y_test: True test labels
        y_pred: Predicted test labels
        num_samples: Number of samples to visualize
    """
    print("Step 6: Visualizing predictions")
    
    # Get indices of correct and incorrect predictions
    correct_indices = np.where(y_test == y_pred)[0]
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    # Function to plot a set of images
    def plot_samples(indices, title_prefix):
        samples = min(num_samples, len(indices))
        if samples == 0:
            print(f"No {title_prefix.lower()} to visualize")
            return
        
        plt.figure(figsize=(15, 3 * samples))
        for i, idx in enumerate(indices[:samples]):
            # Show VH channel (first channel)
            plt.subplot(samples, 2, 2*i+1)
            plt.imshow(X_test[idx, :, :, 0], cmap='viridis')
            plt.title(f'{title_prefix} - VH Channel\nTrue: {"Flood" if y_test[idx] == 1 else "No Flood"}, '
                     f'Pred: {"Flood" if y_pred[idx] == 1 else "No Flood"}')
            plt.colorbar()
            
            # Show VV channel (second channel)
            plt.subplot(samples, 2, 2*i+2)
            plt.imshow(X_test[idx, :, :, 1], cmap='viridis')
            plt.title(f'{title_prefix} - VV Channel\nTrue: {"Flood" if y_test[idx] == 1 else "No Flood"}, '
                     f'Pred: {"Flood" if y_pred[idx] == 1 else "No Flood"}')
            plt.colorbar()
        
        plt.tight_layout()
        filename = os.path.join(RESULTS_DIR, f'{title_prefix.lower().replace(" ", "_")}.png')
        plt.savefig(filename)
        print(f"{title_prefix} visualizations saved to {filename}")
        plt.close()
    
    # Plot correct and incorrect predictions
    plot_samples(correct_indices, "Correct Predictions")
    plot_samples(incorrect_indices, "Incorrect Predictions")

# ====== MAIN FUNCTION ======
def main():
    """Main function to run the entire pipeline."""
    print("====== SAR FLOOD PREDICTION CNN TRAINING ======")
    print(f"Current date and time: {tf.timestamp()}")
    
    try:
        # Step 1: Load dataset
        X_data, y_data = load_dataset()
        
        # Step 2: Split dataset into train, validation, and test sets
        # First split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_data, y_data, 
            test_size=TEST_SPLIT, 
            random_state=RANDOM_SEED, 
            stratify=y_data
        )
        
        # Then split train+val into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=VALIDATION_SPLIT, 
            random_state=RANDOM_SEED, 
            stratify=y_temp
        )
        
        print(f"\nData split complete:")
        print(f"  - Training set: {X_train.shape[0]} images")
        print(f"  - Validation set: {X_val.shape[0]} images")
        print(f"  - Test set: {X_test.shape[0]} images")
        
        # Step 3: Train model
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        # Step 4: Plot training history
        plot_training_history(history)
        
        # Step 5: Evaluate model
        y_pred = evaluate_model(model, X_test, y_test)
        
        # Step 6: Visualize predictions
        visualize_predictions(X_test, y_test, y_pred)
        
        print("\n====== TRAINING COMPLETE ======")
        print(f"Model saved to: {MODEL_DIR}")
        print(f"Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

# Run the script if it's executed directly
if __name__ == "__main__":
    main()
