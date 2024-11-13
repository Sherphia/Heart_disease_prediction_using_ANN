# Heart Disease Prediction using Artificial Neural Network (ANN)

This project uses an artificial neural network to predict heart disease presence based on patient health data. We train the model on various activation functions to compare their effectiveness.

## Project Overview
The model is implemented with Python, using libraries such as TensorFlow and Keras for deep learning, and scikit-learn for data preprocessing and evaluation. The trained model predicts the likelihood of heart disease and evaluates its performance on metrics like accuracy, precision, recall, and F1 score.

## Dataset
The model uses the `heart.csv` dataset containing health attributes (e.g., age, blood pressure) and a target column representing heart disease presence. This data should be preprocessed before model training.

## Model Architecture
The neural network architecture is a simple feedforward model with:
- An input layer taking 13 features.
- Two hidden layers with 8 and 14 neurons, respectively, using Leaky ReLU activation for non-linear transformation.
- An output layer with sigmoid activation for binary classification.

We also compare other activation functions (`relu`, `tanh`, `sigmoid`, `leaky_relu`) by dynamically building the model with each activation type.

## Key Files
- `heart.csv`: Dataset file with heart health data.
- `ann_model.py`: Main Python file implementing the model and evaluations.
- `README.md`: Documentation of the project (this file).

## Libraries Used
- TensorFlow & Keras: For building and training the neural network.
- Scikit-learn: For data preprocessing and performance metrics.
- Matplotlib & Seaborn: For plotting metrics and analysis.

## Getting Started

### Prerequisites
To run the project, you will need:
- Python 3.x
- Libraries specified in `requirements.txt` (install with `pip install -r requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/heart-disease-prediction-ann.git
   cd heart-disease-prediction-ann
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
## Running the Model
1. Place the heart.csv dataset in the project directory.
2. Run the model:
    ```bash
    python ann_model.py
3. The program will output:
    - Predictions for test data.
    - Evaluation metrics: accuracy, precision, recall, and F1 score.
    - A plot comparing performance for each activation function.
## Results
  After training, the model evaluates each activation function's impact on performance, displayed in a plot with accuracy, precision, recall, and F1 score.
