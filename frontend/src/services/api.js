export async function predictSolubility(smiles) {
  const res = await fetch("http://localhost:8000/predict/solubility_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });

  if (!res.ok) {
    throw new Error("Solubility prediction failed");
  }

  return res.json();
}

export async function predictShap(smiles) {
  const res = await fetch("http://localhost:8000/predict/shap_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });

  if (!res.ok) {
    throw new Error("SHAP prediction failed");
  }

  return res.json();
}