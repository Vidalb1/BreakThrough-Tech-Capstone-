# Adult Census Income Classification & Model Comparison
📌 Project Overview
This project implements an end-to-end Machine Learning pipeline to predict whether an individual's annual income exceeds $50,000 based on the 1994 U.S. Census dataset. Designed for Community Development Financial Institutions (CDFIs) to streamline income verification, the project compares a tuned Decision Tree Classifier with a deep Feedforward Neural Network built using Keras/TensorFlow.  


💻 Technical Skills & Frameworks Used
Machine Learning Architecture: Supervised Classification (Decision Trees, Feedforward Neural Networks / Multi-Layer Perceptron), Hyperparameter Optimization via GridSearchCV, Confusion Matrix, and Model Evaluation (F1-score, Accuracy).  


Data Engineering & Preprocessing: Feature Dropping, Mean Imputation with Missingness Flags (age_na, hours-per-week_na), StandardScaler, One-Hot Encoding (pd.get_dummies), Class Weight Balancing.  


Libraries & Frameworks: Python, TensorFlow / Keras, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.  


🛠️ Data Preprocessing & Feature Engineering
Feature Selection & Redundancy: Dropped redundant attributes (fnlwgt, education, relationship) while retaining numerical indicators like education-num.  


Missing Value Imputation: Logged missing flags (age_na, hours-per-week_na) and imputed missing continuous values using feature means.  


Encoding & Scaling: One-hot encoded multi-class categorical variables (pd.get_dummies), label-encoded target outputs, and standardized features using StandardScaler for the neural network pipeline.  


Class Imbalance: Handled the 3:1 majority class bias using balanced class weighting (class_weight='balanced') during tree training.  


📊 Model Performance & Results
| Model Architecture | Test Accuracy | F1-Score | Key Highlights |
| :--- | :---: | :---: | :--- |
| **Decision Tree** (Tuned via GridSearch) | **80.98%** | **0.615** | High interpretability; top features: marital status, age, education. |
| **Deep Neural Network** (4 Dense Layers) | **83.45%** | **0.624** | Highest accuracy; architecture: 64 → 32 → 16 → 8 units (ReLU) + Sigmoid. | 

💡 Key Findings & Recommendations
Feature Importance: Analysis revealed that marital-status_Married-civ-spouse (~23.8%), age (~18.7%), and education-num (~13.3%) were the strongest indicators of income level.  


Production Recommendation: The Decision Tree is recommended for real-world deployment due to its ease of auditability and interpretability when explaining eligibility decisions to non-technical stakeholders.  


🧠 What I Learned

While the Neural Network performed better, its complexity might not be best suited for the business objective. Decision Tree model could be better suited for use by stakeholders and anyone with a non-technical background. 

🔮 Possible Next Steps & Future Work

Perform more testing and experiment with more model parameters. Incorporate more libraries and workflows to get a higher F1 score. 

🚀 How to Run
Clone the repository:

Bash
git clone https://github.com/your-username/census-income-classification.git
cd census-income-classification
Install dependencies:

Bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
Open the Jupyter Notebook:

Bash
jupyter notebook Capstone.ipynb
