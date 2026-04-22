**AI-Based Credit Risk Assessment Dashboard**
An end-to-end Machine Learning solution designed to predict the probability of loan default. This project transforms raw financial and application data into an interactive, explainable dashboard for credit analysts.

🔗 **Live Demo**
Check out the live application here: https://creditrisksystem-n5xtdjwvxukbdpby7ze9k4.streamlit.app/

 **Key Features**
Automated Pipeline: Full data cleaning and feature engineering (Family-size adjusted income, debt-to-income proxies).
Advanced Modeling: Trained using Random Forest and Lasso Regression with SMOTE to address class imbalance.
"What-If" Simulation: Real-time risk prediction based on user-inputted applicant data.
Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) to visualize and explain why the model flag an applicant as high or low risk.

**Project Structure**
creditrisksystem/
├── data/
│   └── clean_credit_risk_dataset.csv   # Processed data for dashboard
├── models/
│   ├── credit_risk_rf_model.pkl       # Trained Random Forest Model
│   ├── credit_risk_scaler.pkl         # Feature Scaling artifact
│   └── credit_risk_label_encoders.pkl # Categorical encoders
├── app.py                             # Streamlit Dashboard code
├── requirements.txt                   # Dependency list
└── credit_risk_assessment_project.ipynb # Research & Training Notebook

**Installation & Setup**

Clone the repository:
git clone  https://github.com/aarav11b16123-png/creditrisksystem.git
cd creditrisksystem

Install dependencies:
pip install -r requirements.txt

Run the Dashboard:
streamlit run app.py

**Challenges & Solutions**
Class Imbalance: The dataset had significantly fewer "default" cases. I used SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes and improve the model's recall.
Feature Engineering: Created a custom income_per_family_member feature which proved to be a high-impact predictor in the SHAP analysis.
Explainability: Successfully implemented SHAP in a Streamlit environment, overcoming global state plotting issues to ensure clear visualizations.




**Technologies Used**

**Core Programming & Environment**
Python 3.x: Primary language used for the entire pipeline.
Jupyter Notebooks: Used for initial data exploration, cleaning, and model experimentation.
VS Code: Main IDE for dashboard development and project management.

**Machine Learning & AI**
Scikit-Learn: Used for building the Random Forest and Lasso Regression models, as well as preprocessing (Scaling, Label Encoding).
SHAP (SHapley Additive exPlanations): Implemented for Explainable AI (XAI) to visualize feature impact and model transparency.
Imbalanced-learn (SMOTE): Applied to handle the significant class imbalance in credit default data.

**Data Science Stack**
Pandas: For advanced data manipulation and feature engineering.
NumPy: Used for efficient numerical computations.
Matplotlib & Seaborn: For generating Exploratory Data Analysis (EDA) charts and model performance plots.

**Deployment & DevOps**
Streamlit: The framework used to build the interactive web dashboard.
GitHub: Version control and project hosting.
Streamlit Community Cloud: Used for hosting the live production app.

<img width="1920" height="1080" alt="Screenshot (1371)" src="https://github.com/user-attachments/assets/590f113c-ed6b-4bd6-b0ca-c0efb6a37114" />
<img width="1920" height="1080" alt="Screenshot (1370)" src="https://github.com/user-attachments/assets/fdb25083-ef25-4c70-85b2-ebde17f23dc6" />
<img width="1920" height="1080" alt="Screenshot (1369)" src="https://github.com/user-attachments/assets/2074056e-fea6-4bda-8597-5e449ebfa5dd" />



