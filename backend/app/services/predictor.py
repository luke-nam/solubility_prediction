import os

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

model = joblib.load(
    os.path.join(os.path.dirname(__file__), 
    "../ml_models/solubility_rf_model.joblib")
)
explainer = joblib.load(
    os.path.join(
        os.path.dirname(__file__), 
        "../ml_models/explainer.pkl")
    )

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    descriptor_names = [
        "MaxEStateIndex",
        "MinEStateIndex",
        "qed",
        "SPS",
        "MolWt",
        "MaxPartialCharge",
        "MinPartialCharge",
        "FpDensityMorgan2",
        "BCUT2D_MWHI",
        "BCUT2D_CHGHI",
        "BCUT2D_LOGPHI",
        "BCUT2D_MRHI",
        "AvgIpc",
        "BalabanJ",
        "HallKierAlpha",
        "Ipc",
        "Kappa3",
        "TPSA",
        "FractionCSP3",
        "NumAromaticCarbocycles",
        "NumAromaticRings",
        "NumHAcceptors",
        "NumHDonors",
        "NumHeteroatoms",
        "NumRotatableBonds",
        "Phi",
        "RingCount",
        "MolLogP",
    ]

    descriptor_vals = {
        name: func(mol) 
        for name, func in Descriptors.descList 
        if name in descriptor_names
    }
    return pd.DataFrame([descriptor_vals])

def predict_label(smiles):
    descriptors = compute_descriptors(smiles)
    prediction_class = model.predict(descriptors)[0]
    solubility_class_labels = {0: "Insoluble", 1: "Slightly soluble", 2: "Soluble"}
    return solubility_class_labels[prediction_class]

def predict_probs(smiles):
    descriptors = compute_descriptors(smiles)
    prediction_probs = model.predict_proba(descriptors)[0]
    return prediction_probs.tolist()

def predict_shap(smiles, top_k=5):
    descriptors_df = compute_descriptors(smiles)
    shap_values = explainer.shap_values(descriptors_df)
    predicted_class = model.predict(descriptors_df)[0]
    shap_values_predicted_class = shap_values[0][:, predicted_class]

    shap_values_flat = shap_values[0] if shap_values.ndim > 1 else shap_values

    descriptor_labels = {
        "MolWt": "Molecular Weight", 
        "MolLogP": "LogP", 
        "TPSA": "TPSA",
        "qed": "QED",
        "FractionCSP3": "Fraction sp3 Carbons",
        "NumHAcceptors": "H-Bond Acceptor Count",
        "NumHDonors": "H-Bond Donor Count",
        "RingCount": "Ring Count",
        "FpDensityMorgan2": "Fragment Density",
        "BalabanJ": "Molecular Complexity (BalabanJ)",
        "MaxEStateIndex": "Max E-State Index",
        "MinEStateIndex": "Min E-State Index",
        "Phi": "Phi (Flexibility)",
        "SPS": "Simple Polar Surface"
    }

    if len(descriptors_df.columns) != len(shap_values_flat):
        print(
            f"Error: Mismatch between descriptors ({len(descriptors_df.columns)}) "
            f"and SHAP values ({len(shap_values_flat)})."
        )
        return pd.DataFrame()
    
    shap_df = pd.DataFrame({
        "Feature": descriptors_df.columns,
        "SHAP Value": shap_values_predicted_class,  # Ensure 1D array
        "Feature Value": descriptors_df.iloc[0].values
    })
    shap_df["Feature"] = (
        shap_df["Feature"]
        .map(descriptor_labels)
        .fillna(shap_df["Feature"])
    )
    shap_df["Abs SHAP Value"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values(by="Abs SHAP Value", ascending=False).head(top_k)
    return shap_df
