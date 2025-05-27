export default function ResultDisplay({ data, smiles }) {
  return (
    <div style={{ marginTop: "1rem", padding: "1rem", borderRadius: "6px" }}>
        <h3>Prediction Result</h3>
        <img
            src={`http://localhost:8000/structure?smiles=${encodeURIComponent(smiles)}`}
            alt="Molecule structure"
            width={300}
        />
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}