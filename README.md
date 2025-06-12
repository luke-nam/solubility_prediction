# Molecular Solubility Prediction

This project implements an aqueous solubility prediction and interpretation tool for small molecules from their SMILES. It features real-time descriptor calculation, probabilistic classification, and SHAP-based interpretability to help end users understand the key drivers behind predictions and aid in lead optimization.

## Project Overview
This project aims to:
- Extract meaningful molecular descriptors from SMILES strings.
- Train and evaluate machine learning models to predict aqueous solubility.
- Provide an interpretation of molecular feature contributions to aqueous solubility predictions.


**Input:** Molecular SMILES string

**Output:**
- Predicted solubility class: `Soluble` (LogS < -4), `Slightly soluble` (-4 < LogS < -2), or `Insoluble` (LogS > -2)
- Associated class probability
- SHAP explanation for top 5 most influential features

## Repo Structure
- `notebooks/`: Jupyter notebooks for data processing, modeling, and SHAP analysis
- `backend/`: FastAPI backend to return model prediction results
- `frontend/`: Web app frontend for SMILES input and model result access

### Solubility App Preview 
![Solubility App Preview](/resources/Input.png)
![Solubility App Preview](/resources/Output.png)