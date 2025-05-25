export default function ErrorMessage({ message }) {
  return (
    <div
      style={{
        backgroundColor: "#ffebee",
        color: "#c62828",
        padding: "1rem",
        borderRadius: "4px",
        marginTop: "1rem",
        marginBottom: "1rem",
        width: "100%",
        maxWidth: "600px",
        fontWeight: "medium",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        textAlign: "center"
      }}
    >
      <span style={{ marginRight: "0.5rem", fontSize: "1.2rem" }}>⚠️</span>
      Error: {message}
    </div>
  );
}