import { readFileSync, writeFileSync } from "node:fs";
import { Matrix } from "./matrix";

type ActivationFn = {
  forward: (x: number[]) => number[];
  backward: (x: number) => number;
};

const _sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const sigmoid: ActivationFn = {
  forward: (nums) => nums.map(_sigmoid),
  backward: (x) => {
    const d = _sigmoid(x);
    return d * (1 - d);
  },
};

// Output activation: softmax, paired with cross-entropy.
// `forward` is a placeholder (softmax is applied row-wise in Layer.forward).
// For softmax + cross-entropy the gradient wrt the logits is exactly
// (activation - label), so no extra activation derivative is needed.
const softmax: ActivationFn = {
  forward: function softmax_row(row: number[]): number[] {
    const max = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  },
  backward: () => 1,
};

type SavedNetwork = {
  version: 1;
  layers: { rows: number; cols: number; data: number[][] }[];
};

export class NeuralNetwork {
  layers = [] as Layer[];

  constructor(layer_neuron_count: number[], seed = 32) {
    for (let i = 0; i < layer_neuron_count.length - 1; i++) {
      const input = layer_neuron_count[i];
      const output = layer_neuron_count[i + 1];

      // softmax output for the final layer (cross-entropy), sigmoid for hidden layers
      const activation = i === layer_neuron_count.length - 2 ? softmax : sigmoid;
      this.layers[i] = new Layer(input, output, activation, seed++);
    }
  }

  forward(input: Matrix) {
    let i = input;
    let intermediates = [] as { output: Matrix; raw: Matrix; iwb: Matrix }[];

    for (const layer of this.layers) {
      const x = layer.forward(i);
      i = x.output;
      intermediates.push(x);
    }

    return { final: i, intermediates };
  }

  get_updates(label: Matrix, a: ReturnType<NeuralNetwork["forward"]>) {
    let error = label.add(a.final.multiply_scalar(-1));
    const layer_updates: Matrix[] = [];

    for (let layer_idx = this.layers.length - 1; layer_idx >= 0; layer_idx--) {
      const layer = this.layers[layer_idx];
      const { iwb, raw } = a.intermediates[layer_idx];

      // for every weight in the layer.matrix get the cumulative updates needed to be done
      const updates = Matrix.init(layer.matrix.rows, layer.matrix.cols);
      for (let i = 0; i < layer.matrix.rows; i++)
        for (let j = 0; j < layer.matrix.cols; j++)
          /**
           * dC/dW = dZ / dW * dA/dZ * dC / dA
           * upd = activation from previous layer * derivative(raw_activation) * error
           */
          updates.arr[i][j] = iwb.arr[0][i] * layer.activation.backward(raw.arr[0][j]) * error.arr[0][j];
      layer_updates.push(updates);

      const new_error = Matrix.init(1, layer.matrix.rows);
      // no need to compute for the final row because they correspond to biases
      for (let i = 0; i < layer.matrix.rows - 1; i++)
        for (let j = 0; j < layer.matrix.cols; j++)
          /**
           * dC/dA(L-1) = dZ / dA(L-1) * dA/dZ * dC / dA
           * upd = previous layer weight * derivative(raw_activation) * error
           */
          new_error.arr[0][i] += layer.matrix.arr[i][j] * layer.activation.backward(raw.arr[0][j]) * error.arr[0][j];
      error = new_error;
    }

    return layer_updates;
  }

  // Serialize the architecture (layer shapes) and weights to a JSON file.
  save(filepath: string) {
    const saved: SavedNetwork = { version: 1, layers: this.layers.map((layer) => layer.matrix.to_json()) };
    writeFileSync(filepath, JSON.stringify(saved));
  }

  // Load the architecture and weights from a file, replacing this network's layers.
  // Activations are positional (softmax on the final layer, sigmoid elsewhere),
  // so they are reconstructed from the saved layer shapes.
  load(filepath: string) {
    const saved = JSON.parse(readFileSync(filepath, "utf-8")) as SavedNetwork;
    if (saved.version !== 1 || !Array.isArray(saved.layers) || saved.layers.length === 0)
      throw new Error(`invalid network file: ${filepath}`);

    for (const layer of saved.layers)
      if (layer.data.length !== layer.rows || layer.data[0]?.length !== layer.cols)
        throw new Error(`invalid layer shape in network file: ${filepath}`);

    // layer matrix is (input + 1) x output, the extra row being biases
    const counts = [saved.layers[0].rows - 1, ...saved.layers.map((layer) => layer.cols)];
    const layers = [] as Layer[];
    for (let i = 0; i < saved.layers.length; i++) {
      const activation = i === saved.layers.length - 1 ? softmax : sigmoid;
      const layer = new Layer(counts[i], counts[i + 1], activation, 0);
      layer.matrix = Matrix.from_existing(saved.layers[i].data);
      layers[i] = layer;
    }
    this.layers = layers;
  }
}

export class Layer {
  matrix: Matrix;
  activation: ActivationFn;

  constructor(input: number, output: number, activation: ActivationFn, seed: number) {
    const input_with_bias = input + 1;
    this.matrix = Matrix.from_uniform(input_with_bias, output, seed, 1 / Math.sqrt(input));
    this.activation = activation;
  }

  forward(input: Matrix) {
    const input_with_bias = input.append_column(1);
    const raw = input_with_bias.multiply(this.matrix);
    const output = Matrix.from_existing([this.activation.forward(raw.arr[0])]);
    return { raw, output, iwb: input_with_bias };
  }

  update(changes: Matrix) {
    this.matrix = this.matrix.add(changes);
  }
}
