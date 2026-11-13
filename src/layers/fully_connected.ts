import { Activation1D } from "../activations";
import { Matrix } from "../matrix";
import { Description, Layer, LayerPropogationOutput, Patch, Tensor, Tensor2D } from "../network";

export class FullyConnectedLayer extends Layer<FullyConnectedLayerSerialized> {
  matrix: Matrix;
  activation: Activation1D;

  constructor(
    public readonly input: number,
    public readonly output: number,
    activation: Activation1D,
    seed: number,
  ) {
    super();

    const input_with_bias = input + 1;
    this.matrix = Matrix.from_uniform(input_with_bias, output, seed, 1 / Math.sqrt(input));
    this.activation = activation;
  }

  forward(input: Tensor): LayerPropogationOutput {
    if (input instanceof Tensor2D == false) {
      throw new Error("Input not suitable for fully connected layer.");
    }

    const input_with_bias = input.matrix.append_column(1);
    const raw = input_with_bias.multiply(this.matrix);
    const output = Matrix.from_existing([this.activation.forward(raw.arr[0])]);
    return new FullyConnectedLayerOutput(raw, output, input_with_bias);
  }

  update(changes: Patch) {
    if (changes instanceof FullyConnectedLayerPatch) this.matrix = this.matrix.add(changes.updates);
    else throw new Error("Incorrect class");
  }

  get_updates(error: Tensor, propogation_output: LayerPropogationOutput) {
    if (error instanceof Tensor2D == false || propogation_output instanceof FullyConnectedLayerOutput == false)
      throw new Error("Incorrect classes passed to get_updates");

    const { iwb, raw } = propogation_output;

    const updates = Matrix.init(this.matrix.rows, this.matrix.cols);
    const new_error = Matrix.init(1, this.matrix.rows);

    for (let i = 0; i < this.matrix.rows; i++) {
      for (let j = 0; j < this.matrix.cols; j++) {
        // Compute common factor dA/dZ * dC/dA used in both weight and loss adjustment
        const common = this.activation.backward(raw.arr[0][j]) * error.matrix.arr[0][j];

        /**
         * dC/dW = dZ / dW * dA/dZ * dC / dA
         * upd = activation from previous layer * derivative(raw_activation) * error
         */
        updates.arr[i][j] = iwb.arr[0][i] * common;

        /**
         * No need to compute for the final row because they correspond to biases
         * dC/dA(L-1) = dZ / dA(L-1) * dA/dZ * dC / dA
         * upd = previous layer weight * derivative(raw_activation) * error
         */
        if (i < this.matrix.rows - 1) new_error.arr[0][i] += this.matrix.arr[i][j] * common;
      }
    }

    return { new_error: new Tensor2D(new_error), patch: new FullyConnectedLayerPatch(updates) };
  }

  to_json(): FullyConnectedLayerSerialized {
    return { weights: this.matrix, activation: this.activation.id };
  }

  describe(c: Description) {
    const channels = c.channels ?? 1;
    if (channels != 1) throw new Error("FullyConnectedLayer has input with more than 1 channel");
    if (c.columns != this.input)
      throw new Error(`FullyConnectedLayer has mismatch between input ${this.input} and ${c.columns}`);

    const str = `FullyConnectedLayer (${this.input} x ${this.output} x 1)`;
    const next: Description = { rows: 1, columns: this.output };
    return { str, next };
  }
}

export class FullyConnectedLayerOutput extends LayerPropogationOutput {
  constructor(
    public readonly raw: Matrix,
    public readonly output: Matrix,
    public readonly iwb: Matrix,
  ) {
    super();
  }

  get_output(): Tensor {
    return new Tensor2D(this.output);
  }
}

export class FullyConnectedLayerPatch extends Patch {
  constructor(public readonly updates: Matrix) {
    super();
  }

  add(p: Patch): Patch {
    if (p instanceof FullyConnectedLayerPatch == false) throw new Error("Not FullyConnectedLayerPatch");
    return new FullyConnectedLayerPatch(this.updates.add(p.updates));
  }

  scale(c: number) {
    return new FullyConnectedLayerPatch(this.updates.multiply_scalar(c));
  }
}

export interface FullyConnectedLayerSerialized {
  weights: Matrix;
  activation: string;
}
