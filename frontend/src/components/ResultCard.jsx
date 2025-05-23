import '../styles/ResultCard.css';

export default function ResultCard({ data, smiles }) {
  const { solubility, shap } = data;
  
  // Get prediction directly from the response
  const predictionText = solubility.prediction;
  
  // Determine solubility category color based on prediction
  const categoryColor = predictionText === "Soluble" ? "#4caf50" : "#f44336";
  
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
              </div>
            </div>
          </div>
          
          <div className="right-column">
            <div className="feature-importance">
              <h4>Feature Importance</h4>
              <div className="table-container">
                <table className="shap-table">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>SHAP Value</th>
                      <th>Feature Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedFeatures.map((feature, index) => (
                      <tr key={index} className={index < 3 ? "top-feature" : ""}>
                        <td className="feature-name">{feature.Feature}</td>
                        <td className={`shap-value ${feature.SHAP_Value > 0 ? 'positive' : 'negative'}`}>
                          {feature.SHAP_Value.toFixed(4)}
                        </td>
                        <td className="feature-value">{feature.Feature_Value.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 