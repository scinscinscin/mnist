import fs from "node:fs";
import { Matrix } from "../../src/matrix";
import { Sample, Tensor2D, Tensor3D } from "../../src/network";

export function load_mnist(path = "./datasets/mnist/mnist_train.csv"): Sample[] {
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

    samples.push({ input: new Tensor2D(input), label: new Tensor2D(label) });
  }

  return samples;
}

export function load_mnist_square(path = "./datasets/mnist/mnist_train.csv"): Sample[] {
  const raw = fs.readFileSync(path, "utf8");
  const lines = raw.split(/\r?\n/);

  const samples: Sample[] = [];
  for (const line of lines) {
    if (!line.trim()) continue;

    const parts = line.split(",");
    if (parts.length !== 785) throw new Error(`expected 785 numbers per line, got ${parts.length}`);

    const pixels = parts.slice(1).map((p) => Number(p) / 255);
    const input = Matrix.from_existing(Array.from({ length: 28 }, (_, i) => pixels.slice(i * 28, (i + 1) * 28)));
    const label = Matrix.init(1, 10);
    label.arr[0][Number(parts[0])] = 1;

    samples.push({ input: new Tensor3D([input]), label: new Tensor2D(label) });
  }

  return samples;
}

export function fisher_yates<T>(array: T[], rng: () => number) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}
