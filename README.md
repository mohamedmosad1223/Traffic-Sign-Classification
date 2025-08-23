# Traffic-Sign-Classification
## Overview
This project focuses on classifying German traffic signs using two different models:

1. **Custom CNN** (highest performance)  
2. **MobileNetV2 (Transfer Learning with Data Augmentation)**

The goal was to compare a lightweight custom CNN against a pretrained MobileNetV2 and observe how a simpler network can outperform a more complex model under certain conditions.

---

## Dataset
- **Dataset:** [GTSRB - German Traffic Sign Recognition Benchmark] 
- **Number of images:** 39,209 in 43 classes  
- **Custom CNN preprocessing:** resized to 32×32×3, **no data augmentation applied**  
- **MobileNetV2 preprocessing:** resized to 224×224×3, **with data augmentation applied**  

**Data Augmentation applied for MobileNetV2:**
- Rotation: ±15°  
- Width/Height shift: ±10%  
- Shear: ±10%  
- Zoom: ±10%  
- Horizontal flip  
- Fill mode: nearest  

---

## Models

### 1. Custom CNN
- Input size: 32×32×3  
- Architecture:
  - 3× Conv2D + MaxPooling layers
  - Flatten
  - Dense(256, ReLU)
  - Dropout(0.5)
  - Dense(43, Softmax)  

**Training Details:**  
- Loss: `sparse_categorical_crossentropy`  
- Optimizer: `Adam`  
- Epochs: 10  
- Batch size: 32  
- **No data augmentation applied**  

**Results (on validation/test set):**
- Accuracy: 0.99  
- Macro Average F1-score: 0.98  
- Weighted Average F1-score: 0.99  

**Observation:**  
Custom CNN outperformed MobileNetV2 despite having smaller input size and no augmentation.

---

### 2. MobileNetV2 (Transfer Learning)
- Input size: 224×224×3  
- Base model: MobileNetV2 pretrained on ImageNet, first 10 layers frozen  
- Added head:
  - GlobalAveragePooling2D
  - Dense(128, ReLU)
  - Dropout(0.5)
  - Dense(43, Softmax)  

**Training Details:**  
- Loss: `sparse_categorical_crossentropy`  
- Optimizer: `Adam`  
- Epochs: 10  
- Batch size: 32  
- **Data augmentation applied** (as listed above)  

**Results (on validation/test set):**
- Accuracy: 0.73  
- Macro Average F1-score: 0.65  
- Weighted Average F1-score: 0.72  

**Observation:**  
MobileNetV2 performed well overall but was outperformed by the simpler Custom CNN on this dataset.

---

## Observations
- The **Custom CNN achieved the highest performance**, highlighting that a simple, well-tuned network can surpass a larger pretrained model in certain tasks.  
- MobileNetV2 required data augmentation and larger input size to perform, but still lagged behind the custom CNN.  
- This demonstrates that **simplicity can outperform complexity** when the architecture is well-suited to the data.

---

## Next Steps / Improvements
- Further fine-tuning of MobileNetV2 layers could improve its performance.  
- Introducing minimal augmentation for Custom CNN could further enhance generalization.  
- Ensemble methods could be explored to combine strengths of both models.

---

## Conclusion
This project shows that a **carefully designed custom CNN, even without augmentation, can outperform a pretrained MobileNetV2 with augmentation**, underlining the importance of architecture choice relative to the dataset.
