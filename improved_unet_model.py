# --- 1. SETUP AND IMPORTS ---
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import random

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- 2. DEFINE PATHS (Modified for local environment) ---
BASE_PATH = './task4'  # Adjust this path to your local data directory
TRAIN_IMAGES_DIR = os.path.join(BASE_PATH, 'train_original_processed')
TRAIN_MASKS_DIR = os.path.join(BASE_PATH, 'train_groundtruth_processed')
TEST_IMAGES_DIR = os.path.join(BASE_PATH, 'test_original_processed')
TEST_MASKS_DIR = os.path.join(BASE_PATH, 'test_groundtruth_processed')

# Create directories if they don't exist
os.makedirs(BASE_PATH, exist_ok=True)

# --- 3. IMPROVED DATA LOADING WITH PROPER VALIDATION SPLIT ---
def load_data_pairs(image_dir, mask_dir):
    images, masks, filenames = [], [], []
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Warning: Directories {image_dir} or {mask_dir} don't exist. Creating dummy data for demonstration.")
        # Create dummy data for demonstration
        dummy_images = np.random.rand(100, 256, 256, 1).astype(np.float32)
        dummy_masks = (np.random.rand(100, 256, 256, 1) > 0.5).astype(np.float32)
        return dummy_images, dummy_masks, [f"dummy_{i}.png" for i in range(100)]
    
    image_files = sorted(os.listdir(image_dir))
    for filename in image_files:
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                # Resize to consistent size
                img = cv2.resize(img, (256, 256))
                mask = cv2.resize(mask, (256, 256))
                images.append(np.expand_dims(img, axis=-1))
                masks.append(np.expand_dims(mask, axis=-1))
                filenames.append(filename)
    
    return np.array(images, dtype=np.float32) / 255.0, np.array(masks, dtype=np.float32) / 255.0, filenames

# Load all data
X_all, y_all, filenames = load_data_pairs(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
X_test, y_test, test_filenames = load_data_pairs(TEST_IMAGES_DIR, TEST_MASKS_DIR)

# Proper train-validation split with stratification based on mask content
mask_sums = [np.sum(mask) for mask in y_all]
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=np.digitize(mask_sums, bins=10)
)

print(f"Training images shape: {X_train.shape}")
print(f"Validation images shape: {X_val.shape}")
print(f"Testing images shape: {X_test.shape}")

# --- 4. IMPROVED DATA AUGMENTATION ---
def create_augmented_generator(X, y, batch_size=16, augment=True):
    if augment:
        data_gen_args = dict(
            rotation_range=10,  # Reduced rotation
            width_shift_range=0.05,  # Reduced shift
            height_shift_range=0.05,
            zoom_range=0.1,  # Reduced zoom
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
            brightness_range=[0.9, 1.1],
            rescale=None  # We already normalized
        )
    else:
        data_gen_args = dict(rescale=None)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 42
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed, shuffle=True)
    mask_generator = mask_datagen.flow(y, batch_size=batch_size, seed=seed, shuffle=True)
    
    while True:
        x_batch = next(image_generator)
        y_batch = next(mask_generator)
        # Ensure masks are binary
        y_batch = (y_batch > 0.5).astype(np.float32)
        yield x_batch, y_batch

# --- 5. ENHANCED LOSS FUNCTIONS AND METRICS ---
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def focal_loss(alpha=0.8, gamma=2):
    def focal_loss_with_logits(y_true, y_pred):
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (tf.keras.backend.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(focal_loss)
    return focal_loss_with_logits

def combined_loss(y_true, y_pred):
    # Combination of focal loss, dice loss, and boundary loss
    focal = focal_loss()(y_true, y_pred)
    dice = 1 - dice_coefficient(y_true, y_pred)
    
    # Boundary loss (emphasizes edges)
    kernel = tf.constant([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    edges_true = tf.nn.conv2d(y_true, kernel, strides=[1,1,1,1], padding='SAME')
    edges_pred = tf.nn.conv2d(y_pred, kernel, strides=[1,1,1,1], padding='SAME')
    boundary = tf.keras.backend.mean(tf.keras.backend.square(edges_true - edges_pred))
    
    return 0.5 * focal + 0.3 * dice + 0.2 * boundary

# --- 6. ENHANCED ATTENTION U-NET ARCHITECTURE ---
def squeeze_excite_block(inputs, ratio=16):
    """Squeeze and Excitation block"""
    filters = inputs.shape[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(filters // ratio, activation='relu')(x)
    x = tf.keras.layers.Dense(filters, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((1, 1, filters))(x)
    return tf.keras.layers.multiply([inputs, x])

def residual_conv_block(inputs, num_filters, dropout_rate=0.1):
    """Residual convolution block with squeeze-excite"""
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Squeeze and excite
    x = squeeze_excite_block(x)
    
    # Residual connection
    if inputs.shape[-1] != num_filters:
        residual = Conv2D(num_filters, 1, padding='same')(inputs)
        residual = BatchNormalization()(residual)
    else:
        residual = inputs
    
    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x

def attention_block(F_g, F_l, F_int):
    """Enhanced attention block"""
    g = Conv2D(F_int, (1, 1), padding='same')(F_g)
    g = BatchNormalization()(g)
    
    x = Conv2D(F_int, (1, 1), padding='same')(F_l)
    x = BatchNormalization()(x)
    
    psi = Add()([g, x])
    psi = Activation('relu')(psi)
    psi = Conv2D(1, (1, 1), padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    
    return tf.keras.layers.multiply([F_l, psi])

def create_enhanced_attention_unet(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder with residual blocks
    c1 = residual_conv_block(inputs, 64, 0.05)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = residual_conv_block(p1, 128, 0.05)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = residual_conv_block(p2, 256, 0.1)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = residual_conv_block(p3, 512, 0.1)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = residual_conv_block(p4, 1024, 0.2)

    # Decoder with attention
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    att6 = attention_block(up6, c4, 256)
    merge6 = concatenate([up6, att6])
    c6 = residual_conv_block(merge6, 512, 0.1)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    att7 = attention_block(up7, c3, 128)
    merge7 = concatenate([up7, att7])
    c7 = residual_conv_block(merge7, 256, 0.1)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    att8 = attention_block(up8, c2, 64)
    merge8 = concatenate([up8, att8])
    c8 = residual_conv_block(merge8, 128, 0.05)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    att9 = attention_block(up9, c1, 32)
    merge9 = concatenate([up9, att9])
    c9 = residual_conv_block(merge9, 64, 0.05)

    # Output layer with deep supervision
    outputs = Conv2D(1, 1, activation='sigmoid', name='main_output')(c9)
    
    return Model(inputs=[inputs], outputs=[outputs])

# --- 7. MODEL COMPILATION AND TRAINING ---
input_shape = X_train.shape[1:]
model = create_enhanced_attention_unet(input_shape)

# Use different learning rates for different parts
optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    metrics=[dice_coefficient, iou_coefficient, 'accuracy']
)

model.summary()

# --- 8. TRAINING WITH IMPROVED CALLBACKS ---
print("\n--- Training the Enhanced Attention U-Net Model ---")

BATCH_SIZE = 8  # Reduced batch size for better gradient updates
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "enhanced_attention_unet.h5")

# Create generators
train_generator = create_augmented_generator(X_train, y_train, BATCH_SIZE, augment=True)
val_generator = create_augmented_generator(X_val, y_val, BATCH_SIZE, augment=False)

callbacks = [
    EarlyStopping(
        monitor='val_dice_coefficient', 
        patience=20, 
        verbose=1, 
        mode='max', 
        restore_best_weights=True,
        min_delta=0.001
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH, 
        monitor='val_dice_coefficient', 
        save_best_only=True, 
        mode='max', 
        verbose=1,
        save_weights_only=False
    ),
    ReduceLROnPlateau(
        monitor='val_dice_coefficient', 
        factor=0.5, 
        patience=8, 
        verbose=1, 
        mode='max', 
        min_lr=1e-7,
        cooldown=3
    )
]

# Training
EPOCHS = 150
STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
VAL_STEPS = len(X_val) // BATCH_SIZE

history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=VAL_STEPS,
    callbacks=callbacks,
    verbose=1
)

# --- 9. EVALUATION AND VISUALIZATION ---
print("\n--- Final Evaluation ---")

# Load best model
model.load_weights(MODEL_SAVE_PATH)

# Evaluate on test set
test_loss, test_dice, test_iou, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Dice Coefficient: {test_dice:.4f}")
print(f"Test IoU: {test_iou:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Evaluate on validation set
val_loss, val_dice, val_iou, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Dice Coefficient: {val_dice:.4f}")

# Evaluate on training set (subset to avoid memory issues)
train_subset_indices = np.random.choice(len(X_train), min(len(X_val), len(X_train)), replace=False)
X_train_subset = X_train[train_subset_indices]
y_train_subset = y_train[train_subset_indices]
train_loss, train_dice, train_iou, train_acc = model.evaluate(X_train_subset, y_train_subset, verbose=0)
print(f"Training Dice Coefficient (subset): {train_dice:.4f}")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', alpha=0.8)
plt.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['dice_coefficient'], label='Training Dice Coefficient', alpha=0.8)
plt.plot(history.history['val_dice_coefficient'], label='Validation Dice Coefficient', alpha=0.8)
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['iou_coefficient'], label='Training IoU', alpha=0.8)
plt.plot(history.history['val_iou_coefficient'], label='Validation IoU', alpha=0.8)
plt.title('IoU Coefficient')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- 10. VISUALIZE PREDICTIONS ---
def visualize_predictions(model, X_data, y_data, num_samples=5):
    predictions = model.predict(X_data[:num_samples])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_data[i].squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(y_data[i].squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_binary = (predictions[i].squeeze() > 0.5).astype(np.float32)
        axes[i, 2].imshow(pred_binary, cmap='gray')
        dice_score = dice_coefficient(y_data[i:i+1], predictions[i:i+1]).numpy()
        axes[i, 2].set_title(f'Prediction {i+1}\nDice: {dice_score:.3f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'predictions_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()

print("\n--- Visualizing Predictions ---")
visualize_predictions(model, X_test, y_test, num_samples=5)

print(f"\nModel and visualizations saved to: {BASE_PATH}")
print("Training completed successfully!")