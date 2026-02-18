🌸 Flower Classification using Machine Learning
📌 Project Overview

This project performs Flower Classification using the famous Iris Dataset.
The goal is to classify flowers into different species based on their physical measurements.

The project includes:

		Data loading and preprocessing
		
		Exploratory Data Analysis (EDA)
		
		Feature visualization
		
		Model training (Logistic Regression & Decision Tree)
		
		Model comparison using accuracy
		
		Confusion Matrix visualization
		
		CLI-based prediction for new inputs

📊 Dataset Information

The Iris dataset contains 150 samples of flowers with 4 features:

		Sepal Length (cm)
		
		Sepal Width (cm)
		
		Petal Length (cm)
		
		Petal Width (cm)

Target classes:

		Setosa
		
		Versicolor
		
		Virginica

Dataset Source:
Loaded directly from sklearn.datasets

⚙ Technologies Used

		Python 3.13.12
		
		NumPy
		
		Pandas
		
		Matplotlib
		
		Scikit-learn

🚀 Project Workflow
1️⃣ Data Loading

The Iris dataset is loaded using load_iris() from Scikit-learn.

2️⃣ Exploratory Data Analysis

		Dataset information
		
		Statistical summary
		
		Feature visualization using scatter plots

3️⃣ Model Training

Two classifiers were trained:

		Logistic Regression
		
		Decision Tree Classifier

4️⃣ Model Evaluation

Models were evaluated using:

		Accuracy Score
		
		Confusion Matrix

5️⃣ CLI Prediction Tool

User can input:

		Sepal Length
		
		Sepal Width
		
		Petal Length
		
		Petal Width

The model predicts the flower species.

📈 Model Performance

Both models were compared using accuracy score.

Example Output:

		Logistic Regression Accuracy: ~97%
		
		Decision Tree Accuracy: ~100%

(Actual results may vary slightly.)

🖥 How to Run the Project
Step 1: Install Required Libraries
python -m pip install numpy pandas matplotlib scikit-learn

Step 2: Run the Script
python flower_classification.py

Step 3: Enter Flower Measurements

Example:

Enter Sepal Length: 5.1
Enter Sepal Width: 3.5
Enter Petal Length: 1.4
Enter Petal Width: 0.2


Output:

Predicted Species: setosa

🎯 Learning Outcomes

		Understanding classification algorithms
		
		Comparing multiple ML models
		
		Evaluating model performance
		
		Interpreting confusion matrix
		
		Building simple CLI-based ML applications

📌 Internship Task

This project was completed as part of the Syntecxhub Virtual Internship Program.

👨‍💻 Author

Tushar Sonawane
AIML Student
Machine Learning Intern
