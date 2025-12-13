import fs from "node:fs/promises";
import { Matrix } from "../src/matrix";

async function parseCSV(file: string, encoder: (x: number) => Matrix) {
  const [headers, ...rows] = file
    .split("\n")
    .map((e) => e.trim())
    .filter((e) => e.length > 0);

  return rows.map((row) => {
    const [label, ...rawInputs] = row.split(",").map((value) => parseFloat(value));
    const inputs = Matrix.fromData([rawInputs]);
    return { label: encoder(label), inputs };
  });
}

export async function loadData(filePath: string, encoder: (x: number) => Matrix) {
  const file = await fs.readFile(filePath, "utf-8");
  return await parseCSV(file, encoder);
}
