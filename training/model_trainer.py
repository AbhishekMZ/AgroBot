# training/model_trainer.py
import tensorflow as tf
import os
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

def create_model(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.2):
    """Create a MobileNetV2-based model for transfer learning.
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate to prevent overfitting
        
    Returns:
        Compiled Keras model
    """
    # Create the base model from MobileNetV2
    base_model = MobileNetV2(input_shape=input_shape, 
                            include_top=False, 
                            weights='imagenet')
                            
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Final classification layer
    if num_classes == 2:
        # Binary classification
        predictions = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        # Multi-class classification
        predictions = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    # Assemble the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    return model

def fine_tune_model(model, num_fine_tune_layers=30):
    """Fine-tune the later layers of the base model.
    
    Args:
        model: The pre-trained model
        num_fine_tune_layers: Number of late layers to unfreeze for fine-tuning
        
    Returns:
        Re-compiled model with fine-tuning layers unfrozen
    """
    # Get the base MobileNetV2 model
    base_model = model.layers[0]
    
    # Unfreeze the top layers of the model
    base_model.trainable = True
    
    # Freeze all layers except the last num_fine_tune_layers
    for layer in base_model.layers[:-num_fine_tune_layers]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    if model.output_shape[-1] == 1:  # Binary classification
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:  # Multi-class
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_data_generators(train_dir, validation_dir, batch_size=32, img_size=(224, 224)):
    """Create data generators for training and validation.
    
    Args:
        train_dir: Directory with training images (should have class subdirectories)
        validation_dir: Directory with validation images (should have class subdirectories)
        batch_size: Batch size for training
        img_size: Input image dimensions (height, width)
        
    Returns:
        train_generator, validation_generator, class mapping
    """
    # Apply data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow from directory for training
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Flow from directory for validation
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class indices mapping
    class_indices = train_generator.class_indices
    
    return train_generator, validation_generator, class_indices

def train_model(args):
    """Train the model using provided arguments.
    
    Args:
        args: Command line arguments with training parameters
    """
    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create model directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging directory for TensorBoard
    log_dir = os.path.join(args.output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Create data generators
    train_generator, validation_generator, class_indices = create_data_generators(
        args.train_dir,
        args.validation_dir,
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width)
    )
    
    # Invert class indices to get class names
    class_names = {v: k for k, v in class_indices.items()}
    
    # Save class mapping
    with open(os.path.join(args.output_dir, 'class_indices.txt'), 'w') as f:
        for class_id, class_name in class_names.items():
            f.write(f"{class_id}: {class_name}\n")
    
    # Create the model
    num_classes = len(class_indices)
    print(f"Creating model with {num_classes} classes: {list(class_names.values())}")
    model = create_model(
        input_shape=(args.img_height, args.img_width, 3),
        num_classes=num_classes,
        dropout_rate=args.dropout_rate
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, 'model_checkpoint.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    
    # Initial training phase (only top layers)
    print("Starting initial training phase...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // args.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // args.batch_size,
        epochs=args.initial_epochs,
        callbacks=callbacks
    )
    
    # Save initial model
    model.save(os.path.join(args.output_dir, 'initial_model.h5'))
    
    # Fine-tuning phase
    if args.fine_tune_epochs > 0:
        print("Starting fine-tuning phase...")
        model = fine_tune_model(model, num_fine_tune_layers=args.fine_tune_layers)
        
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // args.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // args.batch_size,
            epochs=args.initial_epochs + args.fine_tune_epochs,
            initial_epoch=args.initial_epochs,
            callbacks=callbacks
        )
    
    # Save final model
    model.save(os.path.join(args.output_dir, 'final_model.h5'))
    
    # Save TFLite model (for later quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(args.output_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    print(f"Training complete. Model saved to {args.output_dir}")
    print(f"Run quantization script next for INT8 conversion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TFLite-compatible model for AgroBot")
    
    # Data parameters
    parser.add_argument('--train_dir', required=True, help='Directory with training data')
    parser.add_argument('--validation_dir', required=True, help='Directory with validation data')
    parser.add_argument('--output_dir', default='models', help='Directory to save models')
    
    # Model parameters
    parser.add_argument('--img_height', type=int, default=224, help='Image height')
    parser.add_argument('--img_width', type=int, default=224, help='Image width')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--initial_epochs', type=int, default=15, help='Initial training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=15, help='Fine-tuning epochs')
    parser.add_argument('--fine_tune_layers', type=int, default=30, help='Number of layers to fine-tune')
    
    args = parser.parse_args()
    
    train_model(args)