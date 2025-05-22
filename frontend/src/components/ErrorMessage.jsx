export default function ErrorMessage({ message }) {
  return (
    <div style={{ color: "red", marginTop: "1rem", fontWeight: "bold" }}>
      ⚠️ Error: {message}
    </div>
  );
}