# 📊 Loan Delinquency Prediction Model For Freddie Mac 

## 🚀 Project Overview  
This project develops a **Loan Delinquency Prediction Model** that estimates the probability of a **30-day delinquent loan** transitioning to:  
- **Current (paid off)**  
- **Remaining 30 days delinquent**  
- **Progressing to 60 or 90 days delinquent**  

The model leverages **historical loan performance data (2000-2018)** to predict delinquency transitions, helping financial institutions assess risk and optimize loan management strategies.  

---

## 🏗️ Key Features  
✔ **Data Pipeline Development:** Processed large-scale loan performance data using **Pandas** and **NumPy**.  
✔ **Feature Engineering:** Extracted key financial indicators such as **loan age, interest rates, repayment history, and zero-balance codes**.  
✔ **Machine Learning Modeling:** Built a **classification model** to predict loan delinquency transitions.  
✔ **Performance Metrics:** Optimized models to improve **precision and recall** for delinquency prediction.  

---

## 📂 Repository Structure  
```
📁 Loan_Delinquency_Prediction/
│── 📂 data/               # Raw and processed datasets
│── 📂 scripts/            # Production-ready scripts for data processing
│    │── data_loading.py          # Loads loan performance data
│    │── data_cleaning.py         # Cleans missing values, standardizes columns
│    │── feature_engineering.py   # Extracts and transforms delinquency trends
│    │── data_merging.py          # Merges cleaned data with historical records
│    │── Model_Training.ipynb        # Train different models (Logistic Regression, Random Forest, XGBoost)
│    │── Model_Evaluation.ipynb      # Evaluate models (Precision, Recall, AUC-ROC, Confusion Matrix)
│── 📂 notebooks/         # Exploratory and model development notebooks
│── 📄 README.md
│── 📄 requirements.txt
```

---

## 🛠️ Tech Stack  
- **Languages:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
- **Machine Learning Models:** Logistic Regression, Random Forest, XGBoost  
- **Evaluation Metrics:** Precision, Recall, F1-score, AUC-ROC 

---

## 📊 Model Performance  
✅ Developed **classification models** to predict loan delinquency transitions  
✅ Extracted **key financial indicators** to improve prediction accuracy  
✅ Achieved **optimized performance** using **feature engineering & model tuning**  

---

## 📌 How to Run the Project  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/Loan_Delinquency_Prediction.git
cd Loan_Delinquency_Prediction
```

### 2️⃣ Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

### 3️⃣ Provide Your Own Data (Due to NDA Restrictions)
This project does not include actual data files. To run the pipeline, you must provide your own dataset.

🔹 Expected File Formats & Naming:

- Loan performance data should be in CSV format (or another format specified in `scripts/data_loading.py`).
- Place your data files in the `data/` directory.
- Refer to `data_instructions.md` for column names and expected formats.

### 4️⃣ Run Data Processing Scripts
Before training the model, preprocess the data using the scripts inside scripts/:

```bash
python scripts/data_loading.py        # Load loan performance data
python scripts/data_cleaning.py       # Clean and preprocess data
python scripts/feature_engineering.py # Transform data for model training
python scripts/data_merging.py        # Merge with historical loan data
```
### 5️⃣ Apply One-Hot Encoding
Convert categorical variables into numerical format for model training:
```bash
python scripts/one_hot_encoding.py
```
### 6️⃣ Train different models (Logistic Regression, Random Forest, XGBoost).  
Run the model training script:
```bash
python scripts/model_training.py
```
### 7️⃣ Evaluate Model Performance (Includes EDA)
Once the model is trained, analyze performance using EDA, metrics, and visualizations:
```bash
python scripts/model_evaluation.py
```

### 🚀 Notes:

- If your data files change, update the file paths in the scripts accordingly.
- You can modify hyperparameters or model selection in `scripts/model_training.py`.
- Once the workflow is stable, consider automating the pipeline using Apache Airflow.
---

## 📢 Future Improvements  

🔹 Implement **deep learning models (LSTMs)** for time-series prediction  
🔹 Integrate **Snowflake** for scalable data processing  
🔹 Develop a **real-time delinquency risk dashboard**  

---

## 📬 Contact  

For any questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/tanmaysakharkar) or open an issue!  
