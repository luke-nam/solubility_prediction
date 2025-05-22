export default function ResultDisplay({ data }) {
  return (
    <div style={{ marginTop: "1rem", padding: "1rem", borderRadius: "6px" }}>
      <h3>Prediction Result</h3>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}