# Natural Images Classification using CNN

A deep learning project that uses a Convolutional Neural Network to classify images into 8 categories.

**Author:** Lisseth Zamora  
**Course:** CECS 456 - Machine Learning  
**Date:** December 17, 2025

---

## Project Overview

This project classifies natural images into 8 categories:
- Airplane, Car, Cat, Dog, Flower, Fruit, Person, Motorbike

**Dataset:** [Natural Images on Kaggle](https://www.kaggle.com/datasets/prasunroy/natural-images)  
**Total Images:** 6,899  
**Model Accuracy:** 92.88%

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Upload `natural_images_cnn_project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: **Runtime → Change runtime type → GPU**
3. Download dataset from [Kaggle](https://www.kaggle.com/datasets/prasunroy/natural-images)
4. Upload the zip file when prompted in the notebook
5. Run all cells (takes ~15-20 minutes)

### Option 2: Local Setup
```bash
# Install requirements
pip install tensorflow keras numpy matplotlib seaborn scikit-learn

# Download dataset from Kaggle and place in project folder
# Open and run the notebook
jupyter notebook natural_images_cnn_project.ipynb
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Results

### Performance
- **Validation Accuracy:** 92.88%
- **Training Accuracy:** 92.14%
- **Best Class:** Fruit (100% accuracy)
- **Most Challenging:** Dog (80% accuracy)
- **Common Confusion:** Cats and dogs (similar features)

### Model Details
- **Architecture:** 4 Convolutional blocks (32→64→128→128 filters)
- **Parameters:** 5,553,864
- **Training Time:** ~20 minutes on GPU

---

## Dataset Distribution

| Class | Number of Images |
|-------|------------------|
| Airplane | 727 |
| Car | 968 |
| Cat | 885 |
| Dog | 702 |
| Flower | 843 |
| Fruit | 1000 |
| Motorbike | 788 |
| Person | 986 |
| **Total** | **6,899** |

---

## Repository Structure

```
natural-images-classification/
├── natural_images_cnn_project.ipynb    # Main project code
├── README.md                           # This file
├── requirements.txt                    # Python packages
└── results/                            # Generated images
    ├── sample_images.png
    ├── training_results.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

---

## Important Notes

- **Dataset NOT included** (350MB - too large for GitHub)
- Download from Kaggle link above
- **Trained model NOT included** (65MB)
- Model will train when you run the notebook

---

## Visualizations

### Sample Images
![Sample Images](results/sample_images.png)

### Training Progress
![Training Results](results/training_results.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Sample Predictions
![Sample Predictions](results/sample_predictions.png)

---

## Key Findings

**What Worked:**
- Data augmentation prevented overfitting
- Dropout (50%) improved generalization
- CNN learned distinctive features for each class

**Challenges:**
- Cats/dogs frequently confused (similar body structure, fur)
- Person class has high variation (poses, clothing)

**Future Improvements:**
- Use transfer learning (ResNet, MobileNet)
- Collect more cat/dog images
- Try different architectures

---

## Technologies Used

- **TensorFlow/Keras** - Deep learning
- **NumPy** - Numerical operations
- **Matplotlib/Seaborn** - Visualizations
- **Scikit-learn** - Metrics
- **Google Colab** - GPU training

---

## Contact

**Lisseth Zamora**  
CECS 456 - Machine Learning  
California State University, Long Beach
