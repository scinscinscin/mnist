import { Activation1D } from "../activations";
import { Matrix } from "../matrix";
import { Description, Layer, LayerPropogationOutput, Patch, Tensor, Tensor2D, Tensor3D } from "../network";

export class SequentialFullyConnectedLayer extends Layer<SequentialFullyConnectedLayerSerialized> {
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
    if (input instanceof Tensor3D == false) throw new Error("Input not suitable for SequentialFullyConnectedLayer.");
    let rows: number[][] = input.matrix.map((step) => step.arr[0]);

    const { raw, output, iwb } = this.compute_steps(rows);
    return new SequentialFullyConnectedLayerOutput(raw, output, iwb);
  }

  // For each input row [x, 1] compute its raw projection, its activated output,
  // and the bias-augmented input. Shared across every time step.
  private compute_steps(rows: number[][]) {
    const raw = [] as Matrix[];
    const output = [] as Matrix[];
    const iwb = [] as Matrix[];

    for (const row of rows) {
      const input_with_bias = Matrix.from_existing([[...row, 1]]);
      const raw_step = input_with_bias.multiply(this.matrix);
      const output_step = Matrix.from_existing([this.activation.forward(raw_step.arr[0])]);

      iwb.push(input_with_bias);
      raw.push(raw_step);
      output.push(output_step);
    }

    return { raw, output, iwb };
  }

  update(changes: Patch) {
    if (changes instanceof SequentialFullyConnectedLayerPatch) this.matrix = this.matrix.add(changes.updates);
    else throw new Error("Incorrect class");
  }

  get_updates(error: Tensor, propogation_output: LayerPropogationOutput) {
    if (propogation_output instanceof SequentialFullyConnectedLayerOutput == false)
      throw new Error("Incorrect classes passed to get_updates");

    const { raw, iwb } = propogation_output;
    const num_steps = raw.length;

    let error_rows: number[][];
    if (error instanceof Tensor3D == false) throw new Error("Expected a Tensor3D error for a sequence input");
    error_rows = error.matrix.map((step) => step.arr[0]);
    if (error_rows.length !== num_steps)
      throw new Error(`SequentialFullyConnectedLayer expected ${num_steps} error steps, got ${error_rows.length}`);

    const updates = Matrix.init(this.matrix.rows, this.matrix.cols);
    const new_error_steps = new Array(num_steps) as Matrix[];

    for (let step = 0; step < num_steps; step++) {
      const raw_row = raw[step].arr[0];
      const iwb_row = iwb[step].arr[0];
      const error_row = error_rows[step];
      const new_error = Matrix.init(1, this.matrix.rows - 1).arr[0];

      for (let i = 0; i < this.matrix.rows; i++) {
        for (let j = 0; j < this.matrix.cols; j++) {
          // Compute common factor dA/dZ * dC/dA used in both weight and loss adjustment
          const common = this.activation.backward(raw_row[j]) * error_row[j];

          /**
           * dC/dW = dZ / dW * dA / dZ * dC / dA
           * upd = activation from previous layer * derivative(raw_activation) * error
           * Accumulated across time steps because the weights are shared.
           */
          updates.arr[i][j] += iwb_row[i] * common;

          /**
           * No need to compute for the final row because they correspond to biases
           * dC/dA(L-1) = dZ / dA(L-1) * dA / dZ * dC / dA
           * upd = previous layer weight * derivative(raw_activation) * error
           */
          if (i < this.matrix.rows - 1) new_error[i] += this.matrix.arr[i][j] * common;
        }
      }

      new_error_steps[step] = Matrix.from_existing([new_error]);
    }

    const new_error = new Tensor3D(new_error_steps);
    return { new_error, patch: new SequentialFullyConnectedLayerPatch(updates) };
  }

  to_json(): SequentialFullyConnectedLayerSerialized {
    return { weights: this.matrix, activation: this.activation.id };
  }

  describe(c: Description) {
    const channels = c.channels ?? 1;
    if (c.columns != this.input)
      throw new Error(`SequentialFullyConnectedLayer has mismatch between input ${this.input} and ${c.columns}`);

    const next: Description =
      channels > 1 ? { rows: 1, columns: this.output, channels } : { rows: 1, columns: this.output };
    const str = `SequentialFullyConnectedLayer (${this.input} x ${this.output} x ${channels})`;
    return { str, next };
  }
}

export class SequentialFullyConnectedLayerOutput extends LayerPropogationOutput {
  constructor(
    public readonly raw: Matrix[], // 1 x output per time step
    public readonly output: Matrix[], // 1 x output per time step
    public readonly iwb: Matrix[], // 1 x (input + 1) per time step
  ) {
    super();
  }

  get_output(): Tensor {
    return new Tensor3D(this.output);
  }
}

export class SequentialFullyConnectedLayerPatch extends Patch {
  constructor(public readonly updates: Matrix) {
    super();
  }

  add(p: Patch): Patch {
    if (p instanceof SequentialFullyConnectedLayerPatch == false)
      throw new Error("Not SequentialFullyConnectedLayerPatch");
    return new SequentialFullyConnectedLayerPatch(this.updates.add(p.updates));
  }

  scale(c: number) {
    return new SequentialFullyConnectedLayerPatch(this.updates.multiply_scalar(c));
  }
}

export interface SequentialFullyConnectedLayerSerialized {
  weights: Matrix;
  activation: string;
}
