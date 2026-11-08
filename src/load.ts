import fs from "node:fs";
import { Matrix } from "./matrix";

export type Sample = {
  input: Matrix;
  label: Matrix;
};

export function load_mnist(path = "./mnist_train.csv"): Sample[] {
  const raw = fs.readFileSync(path, "utf8");
  const lines = raw.split(/\r?\n/);

  const samples: Sample[] = [];
  for (const line of lines) {
    if (!line.trim()) continue;

    const parts = line.split(",");
    if (parts.length !== 785) throw new Error(`expected 785 numbers per line, got ${parts.length}`);

    const input = Matrix.from_existing([parts.slice(1).map((p) => Number(p) / 255)]);
    const label = Matrix.init(1, 10);
    label.arr[0][Number(parts[0])] = 1;

    samples.push({ input, label });
  }

  return samples;
}
