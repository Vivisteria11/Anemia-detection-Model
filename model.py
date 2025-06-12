import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    "data_dir": r'C:\Users\ankit\OneDrive\Desktop\anemia_detection\Conjuctiva',  # Directory containing train/test/val folders
    "input_shape": (224, 224, 3),  # MobileNetV2 default input size
    "batch_size": 32,               # Optimal for your GPU memory
    "num_classes": 2,               # Assuming binary classification (anemic/non-anemic)
    "learning_rate": 0.0001,         # Lower learning rate for fine-tuning
    "epochs": 50,                   # Enough for convergence
    "patience": 10,                  # Early stopping patience
    "freeze_layers": 120,            # Number of layers to freeze (balance between speed and accuracy)
    "tflite_save_path": "anemia_detection.tflite"
}

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Data augmentation and preprocessing
def create_datagen(train=True):
    if train:
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        return ImageDataGenerator(rescale=1./255)

# Create data generators
train_datagen = create_datagen(train=True)
val_datagen = create_datagen(train=False)
test_datagen = create_datagen(train=False)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    os.path.join(CONFIG["data_dir"], "train"),
    target_size=CONFIG["input_shape"][:2],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(CONFIG["data_dir"], "val"),
    target_size=CONFIG["input_shape"][:2],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(CONFIG["data_dir"], "test"),
    target_size=CONFIG["input_shape"][:2],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False
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
        alpha=1.0  # Default width multiplier
    )
    
    # Freeze layers
    for layer in base_model.layers[:CONFIG["freeze_layers"]]:
        layer.trainable = False
    for layer in base_model.layers[CONFIG["freeze_layers"]:]:
        layer.trainable = True
    
    # Add custom top layers
    inputs = tf.keras.Input(shape=CONFIG["input_shape"])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(CONFIG["num_classes"], activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

model = create_model()
model.summary()

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=CONFIG["patience"],
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // CONFIG["batch_size"],
    validation_data=val_generator,
    validation_steps=val_generator.samples // CONFIG["batch_size"],
    epochs=CONFIG["epochs"],
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save the final model
model.save('final_model.h5')

# Convert to TFLite
def convert_to_tflite(model_path, tflite_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply optimizations
    
    # For further optimization (optional)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_path}")

# Convert the best model to TFLite
convert_to_tflite('best_model.h5', CONFIG["tflite_save_path"])

# Test TFLite model (optional)
def test_tflite_model(tflite_path, test_images):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test with sample images
    for i, (image, label) in enumerate(test_images):
        if i >= 5:  # Test with 5 samples
            break
            
        # Preprocess image
        input_data = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"Sample {i+1}:")
        print("True Label:", label)
        print("Predicted Probabilities:", output_data)
        print("Predicted Class:", np.argmax(output_data))
        print()

# To use the test function:
# test_images = [(x, y) for x, y in test_generator][0]  # Get first batch
# test_tflite_model(CONFIG["tflite_save_path"], test_images)
