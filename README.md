# Lithology Prediction from Well Logs with Streamlit and TensorFlow

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [Code Explanation](#code-explanation)
6. [Contributing](#contributing)
7. [License](#license)

---

## Introduction 🌟
This project is a web application built using Streamlit that predicts lithology from well logs. The application uses a pre-trained Convolutional Neural Network (CNN) model from TensorFlow to make predictions. The well logs are visualized using Matplotlib, and the results can be downloaded as a CSV file.

---

## Data Source 📊
The training data used for the pre-trained model comes from the Force-2020-Machine-Learning-competition by bolgebrygg. For more details, you can visit their [GitHub repository](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition).

---


## Requirements 📋
- Python 3.x
- Streamlit
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Joblib

---

## Installation 🛠️

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-repo/lithology-prediction.git
    ```
2. **Navigate to the project directory**
    ```bash
    cd lithology-prediction
    ```
3. **Install the required packages**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run 🚀

1. **Navigate to the project directory**
    ```bash
    cd lithology-prediction
    ```
2. **Run the Streamlit app**
    ```bash
    streamlit run app.py
    ```

---

## Code Explanation 📝

### Importing Libraries
- **Streamlit**: For creating the web application.
- **Pandas**: For data manipulation.
- **TensorFlow**: For loading and using the pre-trained model.
- **Joblib**: For loading the scaler.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.

### Lithology Keys
- A dictionary mapping numerical labels to lithology types.

### Streamlit UI
- Instructions, file uploader, and sliders are created using Streamlit.

### Data Processing and Prediction
- The `predict_lithology()` function reads the uploaded CSV file, processes the data, and makes a prediction using the CNN model.

### Main Function
- Orchestrates the entire flow of the application, from UI creation to data processing and visualization.

---

## Contributing 🤝
Feel free to open issues and pull requests!

---

## License 📜
This project is licensed under the MIT License.

