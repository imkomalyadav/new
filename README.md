# Enhanced Attention U-Net for Image Segmentation

This implementation addresses the issue where validation dice coefficient was higher than training dice coefficient, while also improving overall accuracy.

## Key Improvements Made

### 1. **Fixed Data Leakage Issues**
- **Proper Train-Validation Split**: Implemented stratified splitting based on mask content to ensure balanced distribution
- **Separate Data Loading**: Training and validation data are now properly separated
- **Consistent Preprocessing**: All images are resized to 256x256 for consistency

### 2. **Enhanced Model Architecture**
- **Residual Blocks**: Added residual connections to improve gradient flow
- **Squeeze-and-Excitation**: Improved feature recalibration
- **Enhanced Attention Mechanism**: Better attention blocks with batch normalization
- **Dropout Regularization**: Progressive dropout rates to prevent overfitting

### 3. **Advanced Loss Functions**
- **Combined Loss**: Focal Loss + Dice Loss + Boundary Loss
- **Focal Loss**: Better handling of class imbalance
- **Boundary Loss**: Emphasis on edge detection accuracy
- **Multiple Metrics**: Dice coefficient, IoU, and accuracy tracking

### 4. **Improved Training Strategy**
- **Reduced Batch Size**: Better gradient updates (8 vs 16)
- **Conservative Data Augmentation**: Reduced augmentation intensity to prevent overfitting
- **Better Callbacks**: Improved early stopping and learning rate scheduling
- **Seed Setting**: Reproducible results

### 5. **Why Validation Was Higher Than Training**
The original issue occurred due to:
- **Data Leakage**: Validation and training sets might have overlapped
- **Aggressive Augmentation**: Too much augmentation made training harder
- **Dropout During Training**: Dropout was active during training but not validation
- **Improper Data Splitting**: Random splits without stratification

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Your Data**: Place your data in the following structure:
```
task4/
├── train_original_processed/
├── train_groundtruth_processed/
├── test_original_processed/
└── test_groundtruth_processed/
```

2. **Run the Model**:
```bash
python improved_unet_model.py
```

3. **Adjust Paths**: Modify the `BASE_PATH` variable in the script to point to your data directory.

## Expected Results

With these improvements, you should see:
- **Training Dice > Validation Dice**: Normal learning pattern
- **Higher Overall Accuracy**: Improved segmentation performance
- **Stable Training**: Consistent convergence without overfitting
- **Better Generalization**: More robust performance on test data

## Model Features

### Architecture Enhancements
- **Residual U-Net**: Skip connections within blocks
- **Attention Mechanism**: Focus on relevant features
- **Batch Normalization**: Stable training
- **Progressive Dropout**: Regularization without over-suppression

### Training Improvements
- **Stratified Splitting**: Balanced train/validation sets
- **Multi-component Loss**: Comprehensive optimization
- **Adaptive Learning Rate**: Optimal convergence
- **Early Stopping**: Prevent overfitting

## Files Generated

After training, the following files will be created:
- `enhanced_attention_unet.h5`: Best model weights
- `training_history.png`: Training curves visualization
- `predictions_visualization.png`: Sample predictions

## Troubleshooting

### If validation dice is still higher than training:
1. Check for data leakage in your dataset
2. Reduce augmentation further
3. Increase dropout rates
4. Verify proper train/validation split

### If accuracy is low:
1. Increase training epochs
2. Adjust learning rate
3. Modify loss function weights
4. Check data quality and preprocessing

## Technical Details

### Loss Function Components
- **Focal Loss (50%)**: Handles class imbalance
- **Dice Loss (30%)**: Segmentation-specific metric
- **Boundary Loss (20%)**: Edge preservation

### Regularization Techniques
- Progressive dropout: 0.05 → 0.1 → 0.2
- Batch normalization after each convolution
- L2 regularization via Adam optimizer

### Data Augmentation (Conservative)
- Rotation: ±10 degrees
- Shift: ±5%
- Zoom: ±10%
- Horizontal/Vertical flip
- Brightness: ±10%

This implementation should resolve the validation > training issue while significantly improving segmentation accuracy.
