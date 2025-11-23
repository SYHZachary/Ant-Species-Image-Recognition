# Ant Species Image Recognition

This project classifies ant species from images using deep learning models.

---

## Project Structure

- `datasets/` - labeled images for training/testing  
- `models/` - trained `.h5` models  
- `model_building.ipynb` - Jupyter notebook for training and evaluating models  
- `deploymentUI.py` - Streamlit app for uploading images and predicting species  

---

## How to Run

### 1. Install Dependencies

Make sure you have Python 3.7+ installed, then install required packages:

```bash
pip install tensorflow streamlit pillow matplotlib numpy
```
### 2. Run the Streamlit App

From the project root:
```bash
streamlit run deploymentUI.py
```
- Upload images in JPG, JPEG, or PNG format
- Select a model from the dropdown menu
- View predictions and Grad-CAM heatmaps
