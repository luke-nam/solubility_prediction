import { useState } from "react";
import { predictSolubility, predictShap } from "../services/api";
import SmilesForm from "../components/SmilesForm";
import ResultDisplay from "../components/ResultDisplay";
import ErrorMessage from "../components/ErrorMessage";
import "../styles/Home.css"

export default function Home() {
  const [result, setResult] = useState(null);

  const handlePredict = async (smiles) => {
    try {
        const [solubilityResult, shapResult] = await Promise.all([
          predictSolubility(smiles),
          predictShap(smiles),
        ]);
        
        const result = {
          solubility: solubilityResult,
          shap: shapResult,
        }
        setResult(result);
    } catch (err) {
      setResult({ error: err.message });
    }
  };

  return (
    <div>
      <h1>Solubility Predictor</h1>
      <SmilesForm onSubmit={handlePredict} />
      {result?.error && <ErrorMessage message={result.error} />}
      {result && !result.error && <ResultDisplay data={result} />}
    </div>
  );
}