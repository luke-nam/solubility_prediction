import { useState } from "react";
import { predictSolubility, predictShap } from "../services/api";
import SmilesForm from "../components/SmilesForm";
import ResultCard from "../components/ResultCard";
import ErrorMessage from "../components/ErrorMessage";
import "../styles/Home.css"

export default function Home() {
  const [result, setResult] = useState(null);
  const [smiles, setSmiles] = useState("");

  const handlePredict = async (smilesInput) => {
    try {
        const [solubilityResult, shapResult] = await Promise.all([
          predictSolubility(smilesInput),
          predictShap(smilesInput),
        ]);
        
        const result = {
          solubility: solubilityResult,
          shap: shapResult,
        }
        setSmiles(smilesInput)
        setResult(result);
    } catch (err) {
      setResult({ error: err.message });
    }
  };

  return (
    <div className="home-container">
      <h1>Solubility Predictor</h1>
      <SmilesForm onSubmit={handlePredict} />
      {result?.error && <ErrorMessage message={result.error} />}
      {result && !result.error && <ResultCard data={result} smiles={smiles} />}
    </div>
  );
}