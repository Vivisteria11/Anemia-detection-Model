import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    "data_dir": r'C:\Users\ankit\OneDrive\Desktop\anemia detection\Conjuctiva',
    "input_shape": (224, 224, 3),
    "batch_size": 32,
    "num_classes": 2,
    "learning_rate": 0.00001, # Learning rate for fine-tuning
    "epochs": 50,
    "patience": 10, # Increased patience for early stopping
    "fine_tune_at": -20, # Number of layers from the end to unfreeze (e.g., -20 means unfreeze last 20 layers)
    "dropout_rate": 0.5, # Dropout rate
    "l2_reg": 0.001, # L2 regularization to dense layers
    "tflite_save_path": "anemia_detection.tflite"
}

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Data augmentation and preprocessing
def create_datagen(train=True):
    if train:
        return ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30, # rotation
            width_shift_range=0.25, # width shift
            height_shift_range=0.25, # height shift
            shear_range=0.25, # shear
            zoom_range=0.25, # zoom
            horizontal_flip=True, # horizontal flip
            vertical_flip=True, # vertical flip for more augmentation
            fill_mode='nearest'
        )
    else:
        return ImageDataGenerator(preprocessing_function=preprocess_input)

# Create data generators
train_datagen = create_datagen(train=True)
val_datagen = create_datagen(train=False) # No augmentation for validation
test_datagen = create_datagen(train=False) # No augmentation for testing

# Load datasets
train_generator = train_datagen.flow_from_directory(
    os.path.join(CONFIG["data_dir"], "Training"),
    target_size=CONFIG["input_shape"][:2],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(CONFIG["data_dir"], "Validation"),
    target_size=CONFIG["input_shape"][:2],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False # Keep shuffle as False for consistent validation evaluation
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(CONFIG["data_dir"], "Testing"),
    target_size=CONFIG["input_shape"][:2],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False # Keep shuffle as False for consistent testing evaluation
)

# Verify class indices
print("Class indices:", train_generator.class_indices)

# Create MobileNetV2 model
def create_model():
    # Load pre-trained MobileNetV2 without top
    base_model = MobileNetV2(
        input_shape=CONFIG["input_shape"],
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )

    # Freeze all layers initially for feature extraction
    base_model.trainable = False

    # Add custom top layers
    inputs = tf.keras.Input(shape=CONFIG["input_shape"])
    x = base_model(inputs, training=False) # Ensure base_model runs in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x) 
    # Added L2 regularization to the dense layer
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(CONFIG["l2_reg"]))(x)
    x = layers.Dropout(CONFIG["dropout_rate"])(x) # Use configurable dropout rate
    outputs = layers.Dense(CONFIG["num_classes"], activation='softmax')(x)

    model = models.Model(inputs, outputs)

    # Attach base_model as an attribute for easy access
    model.base_model = base_model

    # Compile the model for the first phase (training only the top layers)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"] * 10), # Higher LR for new layers
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    return model

model = create_model()
model.summary()

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=CONFIG["patience"],
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5, 
    min_lr=1e-9, # Minimum learning rate
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Calculate class weights for imbalanced data
train_labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weights)

# First phase of training: Train only the top layers
print("\n--- Starting Phase 1: Training top layers ---")
history_phase1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // CONFIG["batch_size"],
    validation_data=val_generator,
    validation_steps=val_generator.samples // CONFIG["batch_size"],
    epochs=15, # Increased initial epochs slightly for top layers
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1,
    class_weight=class_weights
)

# Load the best model from phase 1 to continue fine-tuning
model.load_weights('best_model.h5')

# Second phase of training: Fine-tuning a portion of the base model
print("\n--- Starting Phase 2: Fine-tuning a portion of the base model ---")
base_model = model.base_model

# Unfreeze a smaller, more specific portion of the base model
base_model.trainable = True # Unfreeze the entire base model first

# Freeze all layers up to 'fine_tune_at' index
for layer in base_model.layers[:CONFIG["fine_tune_at"]]:
    layer.trainable = False

# Recompile the model with a very low learning rate for fine-tuning
model.compile(
    optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary() 

history_phase2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // CONFIG["batch_size"],
    validation_data=val_generator,
    validation_steps=val_generator.samples // CONFIG["batch_size"],
    epochs=CONFIG["epochs"], # Continue for the total configured epochs
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1,
    class_weight=class_weights
)

# Combine histories for plotting
history_combined = {}
# Ensure histories are combined correctly even if one phase has more epochs due to early stopping
num_epochs_phase1 = len(history_phase1.history['loss'])
num_epochs_phase2 = len(history_phase2.history['loss'])

for key in history_phase1.history.keys():
    history_combined[key] = history_phase1.history[key] + history_phase2.history[key]

# Plot training history
def plot_history(history):
    plt.figure(figsize=(18, 5))

    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # AUC plot
    plt.subplot(1, 3, 3)
    plt.plot(history['auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    # Precision plot
    plt.subplot(1, 2, 1)
    plt.plot(history['precision'], label='Train Precision')
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Recall plot
    plt.subplot(1, 2, 2)
    plt.plot(history['recall'], label='Train Recall')
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig('training_history.png')
    plt.close()

plot_history(history_combined)

# Evaluate on test set
print("Loading best model for evaluation...")
best_model = tf.keras.models.load_model('best_model.h5')
print("Evaluating on test set...")
test_loss, test_acc, test_auc, test_precision, test_recall = best_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Save the final model
best_model.save('final_model.h5')

# Convert to TFLite
def convert_to_tflite(model_path, tflite_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {tflite_path}")

# Convert the best model to TFLite
convert_to_tflite('best_model.h5', CONFIG["tflite_save_path"])
