# Time Studied vs Score Prediction using Linear Regression

## Overview
This project implements a **Linear Regression model** to predict a student's **exam score** based on the number of **hours studied**. The model is trained on a small dataset and visualized using Matplotlib.

## Features
- Uses **Linear Regression** from `scikit-learn` for prediction.
- Splits data into **training** and **testing** sets using `train_test_split`.
- Evaluates the model's accuracy with **R² score**.
- Generates **predictions** for new inputs.
- Visualizes the regression line using **Matplotlib**.

## Dataset
The dataset consists of:
- **Feature (X):** Hours studied
- **Target (y):** Exam score

### Sample Data:
| Hours Studied | Score |
|--------------|-------|
| 70           | 75    |
| 40           | 49    |
| 80           | 91    |
| 50           | 51    |
| 70           | 74    |
| 60           | 68    |
| 20           | 20    |
| 10           | 8     |
| 80           | 87    |
| 60           | 74    |

## Installation
### Prerequisites
Ensure you have **Python** and the following libraries installed:
```bash
pip install numpy matplotlib scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/your-username/time-studied-vs-score.git
cd time-studied-vs-score
```

## Usage
### Running the Model
Execute `main.py` to train the model and make predictions:
```bash
python main.py
```

### Expected Output
```
R² Score: 0.85
Predicted Score for 40 hours studied: 50.3
```

## Code Example
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data
time_studied = np.array([70, 40, 80, 50, 70, 60, 20, 10, 80, 60, 40, 90, 50, 30, 30]).reshape(-1, 1)
scores = np.array([75, 49, 91, 51, 74, 68, 20, 8, 87, 74, 50, 96, 58, 32, 33]).reshape(-1, 1)

# Train-test split
time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(time_train, score_train)

# Evaluation
print(f'R² Score: {model.score(time_test, score_test)}')
print(f'Predicted Score for 40 hours studied: {model.predict(np.array([40]).reshape(-1, 1))[0][0]}')

# Plot Regression Line
plt.scatter(time_studied, scores, label='Actual Data')
plt.plot(np.linspace(0, 90, 100).reshape(-1, 1),
         model.predict(np.linspace(0, 90, 100).reshape(-1, 1)), 'r', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.show()
```

## Project Structure
```
├── main.py              # Linear Regression Model Script
├── README.md            # Project Documentation
```

## Future Improvements
- Add more data points for better accuracy.
- Implement **polynomial regression** for better predictions.
- Deploy as a **web app** using Flask or Streamlit.

## Contributions
Contributions are welcome! Feel free to fork the repo and submit pull requests.

## License
This project is open-source.

