import matplotlib
matplotlib.use("Agg")

import os
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.responses import StreamingResponse
import io



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

def get_shap_influence(smiles, top_k=5):

    descriptor_labels = {
      "MolWt": "Molecular Weight",
      "MolLogP": "LogP",
      "TPSA": "Topological Polar Surface Area",
      "qed": "QED (Drug-likeness)",
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

    descriptors_df = compute_descriptors(smiles)
    shap_values = explainer.shap_values(descriptors_df)
    ordinal_weights = np.array([-1,0,1])
    shap_matrix = shap_values[0]
    influence_scores = np.dot(shap_matrix, ordinal_weights)

    shap_df = pd.DataFrame({
        "Feature": descriptors_df.columns,
        "Influence": influence_scores,
        "Feature_Value": descriptors_df.iloc[0].values
    })
    shap_df["Feature"] = shap_df["Feature"].map(descriptor_labels).fillna(shap_df["Feature"])
    shap_df["Contribution"] = shap_df["Influence"].apply(
        lambda x: "Favors higher solubility class" if x > 0 else "Favors lower solubility class"
    )

    shap_df["Abs_Influence"] = shap_df["Influence"].abs()
    shap_df = shap_df.sort_values(by="Abs_Influence", ascending=False).head(top_k)

    shap_df = shap_df[["Feature", "Feature_Value", "Influence", "Contribution"]]

    return shap_df.to_dict(orient="records")

def display_structure(smiles, width=300, height=300):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, (width, height))
    return img

def display_shap_plot(shap_vals, top_k=5):
    # Limit to top_k features by absolute contribution
    if isinstance(shap_vals, list):
        shap_vals = pd.DataFrame(shap_vals)

    shap_vals = shap_vals.sort_values(by="Influence", ascending=False).head(top_k)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        ax=ax,
        y=[f"{feat} ({val:.2f})" for feat, val in zip(shap_vals["Feature"], shap_vals["Feature_Value"])],
        x=shap_vals["Influence"],
        hue=shap_vals["Contribution"],
        palette={
            "Favors lower solubility class": "#E74C3C",
            "Favors higher solubility class": "#2ECC71"
        },
        dodge=False
    )

    ax.set_title(f"Top {top_k} SHAP Feature Contributions to Predicted Solubility")
    ax.set_xlabel("Directional Influence on Solubility Prediction")
    ax.set_ylabel("Descriptor (Value)")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    # Return image as streaming response
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")