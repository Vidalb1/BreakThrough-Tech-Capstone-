# Adult Census Income Classification & Model Comparison
 
## 📌 Project Overview
 
This project implements an end-to-end machine learning pipeline to predict whether an individual's annual income exceeds $50,000, using the 1994 U.S. Census dataset. The use case is framed around a Community Development Financial Institution (CDFI) that needs a fast, reliable way to estimate income eligibility for its programs. The project follows the full ML life cycle — problem definition, EDA, data preparation, modeling, and evaluation — and compares a tuned Decision Tree Classifier against a deep Feedforward Neural Network (Keras/TensorFlow) to determine which is best suited for deployment.
 
## 🎯 Business Problem
 
- **Dataset:** 1994 U.S. Census demographic and employment data (`censusData.csv`)
- **Label:** `income_binary` — whether an individual earns `<=50K` or `>50K` annually (encoded as 0/1)
- **Features used:** `age`, `education-num`, `occupation`, `workclass`, `marital-status`, `hours-per-week`, `capital-gain`, `capital-loss`, `native-country`, `race`, `sex_selfID` (categoricals one-hot encoded)
- **Why it matters:** A CDFI needs to determine applicant income eligibility for its programs quickly and consistently. A model that predicts income level from readily available demographic and employment features could let the organization triage or pre-screen applicants, informing decisions faster than manual review.
## 💻 Technical Skills & Frameworks Used
 
- **Machine Learning Architecture:** Supervised binary classification (Decision Trees, Feedforward Neural Networks / Multi-Layer Perceptron), hyperparameter optimization via `GridSearchCV`, confusion matrices, and model evaluation (accuracy, F1-score).
- **Data Engineering & Preprocessing:** Feature dropping, mean imputation with missingness flags (`age_na`, `hours-per-week_na`), `StandardScaler`, one-hot encoding (`pd.get_dummies`), label encoding, class weight balancing.
- **Libraries & Frameworks:** Python, TensorFlow/Keras, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.
## 🔍 Exploratory Data Analysis
 
- **Class imbalance:** The label is imbalanced — roughly 3:1 in favor of `<=50K`, visualized with a histogram of `income_binary`.
- **Missing values:** `age`, `workclass`, `occupation`, `hours-per-week`, and `native-country` all contained a meaningful number of `NaN` values.
- **Outliers:** Scatterplots of `capital-gain`, `capital-loss`, `age`, `hours-per-week`, and `education-num` against the label were used to check for outliers; none warranted removal since the downstream model (Decision Tree) does not require winsorizing.
- **Redundant features:** `fnlwgt`, `relationship`, and `education` were dropped — `education` duplicates the information already captured numerically in `education-num`.
## 🛠️ Data Preprocessing & Feature Engineering
 
- **Feature selection & redundancy:** Dropped `fnlwgt`, `education`, and `relationship` to reduce redundancy and multicollinearity.
- **Missing value imputation:** Logged missingness flags (`age_na`, `hours-per-week_na`) before imputing the two numeric columns with missing values (`age`, `hours-per-week`) using their column means.
- **Encoding & scaling:** Label-encoded the target (`income_binary`), one-hot encoded categorical predictors (`pd.get_dummies`, `drop_first=True`), and standardized features with `StandardScaler` for the neural network pipeline only (not needed for the Decision Tree).
- **Class imbalance:** Addressed via `class_weight='balanced'` during Decision Tree training rather than resampling.
## ⚖️ Ethical Considerations
 
Census demographic features carry real risk of encoding bias. `race`, `sex_selfID`, and `native-country` could act as proxies for protected characteristics, and `native-country` is heavily skewed toward the U.S., meaning the model has much less signal for individuals from less-represented countries. If deployed, prediction errors would not be evenly distributed — minority racial groups, non-U.S.-born individuals, and other underrepresented groups are the most likely to be affected by misclassification, since the model has learned patterns dominated by the majority group in the training data. Any real deployment should include a fairness audit across these subgroups before use in an eligibility-screening context.
 
## 📊 Model Performance & Results
 
| Model Architecture | Test Accuracy | F1-Score | Key Highlights |
|---|---|---|---|
| Decision Tree (tuned via `GridSearchCV`, `max_depth=1000`) | 80.98% | 0.615 | High interpretability; top features: marital status, age, education |
| Deep Neural Network (4 hidden layers) | 83.45% | 0.624 | Highest accuracy; architecture: 64 → 32 → 16 → 8 units (ReLU) + sigmoid output |
 
**Neural network configuration:** 4 hidden layers (64/32/16/8 units, ReLU activations), sigmoid output layer, SGD optimizer (`learning_rate=0.1`), binary cross-entropy loss, 100 epochs, 20% validation split.
 
**Training curves:** Training loss steadily decreased while validation loss climbed after roughly epoch 20–30, and training accuracy pulled progressively ahead of validation accuracy — a clear signal of overfitting, though test-set performance still held up reasonably well.
 
## 💡 Key Findings & Recommendations
 
- **Feature importance (Decision Tree):** `marital-status_Married-civ-spouse` (~23.8%), `age` (~18.7%), and `education-num` (~13.3%) were the strongest predictors of income level, followed by `capital-gain` (~10.4%) and `hours-per-week` (~9.4%).
- **Performance comparison:** The neural network outperformed the Decision Tree on both accuracy (+2.5 points) and F1-score (+0.9 points), but the margin is modest given the added complexity.
- **Was the complexity worth it?** For this problem, the performance gain from the neural network is small relative to the extra cost of training, tuning, and maintaining it — and it comes with materially reduced interpretability.
- **Production recommendation:** The Decision Tree is recommended for deployment. It's easier to audit, its feature importances are directly explainable to non-technical stakeholders (e.g., justifying why age or education-num factor into an eligibility screen), and it trains far faster than the neural network — all important when a CDFI needs to explain eligibility decisions to applicants or regulators.
## 🧠 What I Learned
 
The neural network edged out the Decision Tree on both metrics, but the gap wasn't large enough to justify its opacity and tuning overhead for this particular business use case. Simpler, more interpretable models can be the more responsible choice when the people relying on the output need to understand *why* a decision was made — not just that it was accurate.
 
## 🔮 Possible Next Steps & Future Work
 
- Address the neural network's overfitting with dropout, regularization, or early stopping, and experiment with different layer/unit configurations and learning rates.
- Try resampling techniques (e.g., SMOTE) instead of only class weighting to address class imbalance, and compare the effect on F1-score.
- Run a fairness audit of model errors across race, sex, and native-country subgroups before considering any real-world deployment.
- Experiment with additional or alternative features and compare against Logistic Regression and KNN baselines.
## 🚀 How to Run
 
Clone the repository:
```bash
git clone https://github.com/your-username/census-income-classification.git
cd census-income-classification
```
 
Install dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```
 
Open the Jupyter Notebook:
```bash
jupyter notebook Capstone.ipynb
```
 
> Note: Place `censusData.csv` in a `data_capstone/` subdirectory relative to the notebook before running. See the notebook for the full step-by-step analysis, EDA visualizations, and written reflections.
