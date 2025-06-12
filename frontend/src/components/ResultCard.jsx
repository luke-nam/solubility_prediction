import '../styles/ResultCard.css';

export default function ResultCard({ data, smiles }) {
  const { solubility, shap } = data;
  
  // Get prediction directly from the response
  const predictionText = solubility.prediction;
  const max_prob = Math.max(...solubility.probabilities).toFixed(2);

  // Determine solubility category color based on prediction
  const categoryColor = predictionText === "Soluble" ? "#4caf50" : 
                       predictionText === "Slightly soluble" ? "#ffc107" : 
                       "#f44336";
  
  // Sort SHAP values by absolute SHAP value
  const sortedFeatures = [...shap]
    .sort((a, b) => b.Abs_SHAP_Value - a.Abs_SHAP_Value);

  return (
    <div className="result-card">
      <div className="card-header">
        <h3>Solubility Prediction</h3>
      </div>
      
      <div className="card-body">
        <div className="results-grid">
          <div className="structure-section">
            <div className="structure-container">
              <img
                src={`http://localhost:8000/structure?smiles=${encodeURIComponent(smiles)}`}
                alt="Molecule structure"
                className="molecule-image"
              />
              <div className="smiles-string">
                <strong>SMILES:</strong> {smiles}
              </div>
            </div>
            <div className="prediction-result">
              <div className="prediction-category" style={{ backgroundColor: categoryColor }}>
                {predictionText}
                <br />
                Probability: {max_prob}
              </div>
            </div>
          </div>

          <div className="shap-section">
            <div className="shap-container">
              <img
                src={`http://localhost:8000/shap_plot?smiles=${encodeURIComponent(smiles)}`}
                alt="SHAP plot"
                className="shap-image"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 