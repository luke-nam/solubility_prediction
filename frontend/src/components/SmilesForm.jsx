import { useState } from "react";

export default function SmilesForm({ onSubmit }) {
  const [smiles, setSmiles] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(smiles);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="smiles">SMILES:   </label>
      <input
        type="text"
        id="smiles"
        value={smiles}
        onChange={(e) => setSmiles(e.target.value)}
      />
      <button type="submit">Predict</button>
    </form>
  );
}