import { sigmoid, tanh } from "../activations";
import { Matrix } from "../matrix";
import { Description, Layer, LayerPropogationOutput, Patch, Tensor, Tensor3D } from "../network";

// An LSTM processes a whole sequence in one forward pass (backpropagation
// through time in get_updates), so the network's per-sample training loop
// stays unchanged. A sequence of T steps of I features is a Tensor3D of T
// "channels", each a 1 x I row vector; the output is T channels of 1 x H.
//
// Each of the four gates (input, forget, cell candidate, output) owns its own
// weight matrix of shape (input_size + hidden_size + 1) x hidden_size. The
// last row of each matrix is that gate's bias (the fully-connected convention),
// so a gate's pre-activation is a single matmul of the shared row
// [x_t, h_{t-1}, 1] against its own matrix.
export class LSTMLayer extends Layer<LSTMLayerSerialized> {
  input_gate_weights: Matrix;
  forget_gate_weights: Matrix;
  candidate_gate_weights: Matrix;
  output_gate_weights: Matrix;

  constructor(
    public readonly input_size: number,
    public readonly hidden_size: number,
    seed: number,
  ) {
    super();

    const num_gate_inputs = input_size + hidden_size + 1;
    const scale = 1 / Math.sqrt(input_size + hidden_size);
    this.input_gate_weights = Matrix.from_uniform(num_gate_inputs, hidden_size, seed, scale);
    this.forget_gate_weights = Matrix.from_uniform(num_gate_inputs, hidden_size, seed + 1, scale);
    this.candidate_gate_weights = Matrix.from_uniform(num_gate_inputs, hidden_size, seed + 2, scale);
    this.output_gate_weights = Matrix.from_uniform(num_gate_inputs, hidden_size, seed + 3, scale);
  }

  forward(input: Tensor): LSTMLayerOutput {
    if (input instanceof Tensor3D == false) throw new Error("Input not suitable for LSTMLayer.");

    const input_steps = input.matrix;
    if (input_steps.length === 0) throw new Error("LSTMLayer expects at least one time step");

    // make sure that all steps have the same size
    const { rows: step_rows, cols: step_cols } = input_steps[0];
    const matches = input_steps.every((step) => step.rows == step_rows && step.cols == step_cols);
    if (!matches) throw new Error("Input tensor matrices do not have consistent row and col count");
    if (step_rows !== 1) throw new Error(`LSTMLayer expects one row per time step, got ${step_rows}`);
    if (step_cols !== this.input_size)
      throw new Error(`LSTMLayer has mismatch between input ${this.input_size} and ${step_cols}`);

    const num_hidden = this.hidden_size;
    const input_gate_preactivations = [] as Matrix[];
    const forget_gate_preactivations = [] as Matrix[];
    const candidate_gate_preactivations = [] as Matrix[];
    const output_gate_preactivations = [] as Matrix[];
    const hidden_states = [] as Matrix[];
    const cell_states = [] as Matrix[];

    let hidden_state = Matrix.init(1, num_hidden).arr[0];
    let cell_state = Matrix.init(1, num_hidden).arr[0];

    for (const time_step_input of input_steps) {
      const input_row = time_step_input.arr[0];
      // the shared input to every gate at this step: [x_t, h_{t-1}, 1]
      const combined_row = Matrix.from_existing([[...input_row, ...hidden_state, 1]]);

      // each gate is its own linear projection of the combined row, followed
      // by its own activation: input/forget/output use sigmoid, candidate uses
      // tanh
      const forget_gate_preactivation = combined_row.multiply(this.forget_gate_weights).arr[0];
      const input_gate_preactivation = combined_row.multiply(this.input_gate_weights).arr[0];
      const output_gate_preactivation = combined_row.multiply(this.output_gate_weights).arr[0];
      const candidate_gate_preactivation = combined_row.multiply(this.candidate_gate_weights).arr[0];

      forget_gate_preactivations.push(Matrix.from_existing([forget_gate_preactivation]));
      input_gate_preactivations.push(Matrix.from_existing([input_gate_preactivation]));
      output_gate_preactivations.push(Matrix.from_existing([output_gate_preactivation]));
      candidate_gate_preactivations.push(Matrix.from_existing([candidate_gate_preactivation]));

      const forget_gate = sigmoid.forward(forget_gate_preactivation);
      const input_gate = sigmoid.forward(input_gate_preactivation);
      const output_gate = sigmoid.forward(output_gate_preactivation);
      const candidate_gate = tanh.forward(candidate_gate_preactivation);

      // perform element-wise calculations for next cell state and next hidden_state
      const next_cell_state = [] as number[];
      const next_hidden_state = [] as number[];
      for (let unit = 0; unit < num_hidden; unit++) {
        // the next cell state is the current cell state * forget gate + input gate * candidate gate
        next_cell_state[unit] = cell_state[unit] * forget_gate[unit] + input_gate[unit] * candidate_gate[unit];
        // the hidden output of the step is tanh(c_t) gated by the output gate
        next_hidden_state[unit] = Math.tanh(next_cell_state[unit]) * output_gate[unit];
      }

      cell_state = next_cell_state;
      hidden_state = next_hidden_state;

      hidden_states.push(Matrix.from_existing([hidden_state]));
      cell_states.push(Matrix.from_existing([cell_state]));
    }

    return new LSTMLayerOutput(
      input_gate_preactivations,
      forget_gate_preactivations,
      candidate_gate_preactivations,
      output_gate_preactivations,
      hidden_states,
      cell_states,
      input_steps,
    );
  }

  update(changes: Patch) {
    if (changes instanceof LSTMLayerPatch) {
      this.input_gate_weights = this.input_gate_weights.add(changes.input_gate_updates);
      this.forget_gate_weights = this.forget_gate_weights.add(changes.forget_gate_updates);
      this.candidate_gate_weights = this.candidate_gate_weights.add(changes.candidate_gate_updates);
      this.output_gate_weights = this.output_gate_weights.add(changes.output_gate_updates);
    } else throw new Error("Incorrect class");
  }

  get_updates(error: Tensor, propogation_output: LayerPropogationOutput) {
    if (error instanceof Tensor3D == false || propogation_output instanceof LSTMLayerOutput == false)
      throw new Error("Incorrect classes passed to get_updates");

    const {
      input_gate_preactivations,
      forget_gate_preactivations,
      candidate_gate_preactivations,
      output_gate_preactivations,
      hidden_states,
      cell_states,
      input_steps,
    } = propogation_output;

    const num_steps = hidden_states.length;
    if (error.matrix.length !== num_steps)
      throw new Error(`LSTMLayer expected ${num_steps} error steps, got ${error.matrix.length}`);

    const num_hidden = this.hidden_size;
    const num_input = this.input_size;

    const input_gate_updates = Matrix.init(num_input + num_hidden + 1, num_hidden);
    const forget_gate_updates = Matrix.init(num_input + num_hidden + 1, num_hidden);
    const candidate_gate_updates = Matrix.init(num_input + num_hidden + 1, num_hidden);
    const output_gate_updates = Matrix.init(num_input + num_hidden + 1, num_hidden);
    const new_error_steps = new Array(num_steps) as Matrix[];

    // gradients carried back through the recurrence: hidden_carry is
    // dL/dh_{t-1} and cell_carry is f_t * dL/dc_t from the step behind the
    // current one
    let hidden_carry = Matrix.init(1, num_hidden).arr[0];
    let cell_carry = Matrix.init(1, num_hidden).arr[0];

    for (let step_index = num_steps - 1; step_index >= 0; step_index--) {
      const error_step = error.matrix[step_index].arr[0];
      const cell_state = cell_states[step_index].arr[0];
      const previous_cell_state = step_index > 0 ? cell_states[step_index - 1].arr[0] : null;
      const input_row = input_steps[step_index].arr[0];
      const previous_hidden_state = step_index > 0 ? hidden_states[step_index - 1].arr[0] : null;

      // recompute each gate value from its stored pre-activation
      const input_gate = sigmoid.forward(input_gate_preactivations[step_index].arr[0]);
      const forget_gate = sigmoid.forward(forget_gate_preactivations[step_index].arr[0]);
      const candidate_gate = tanh.forward(candidate_gate_preactivations[step_index].arr[0]);
      const output_gate = sigmoid.forward(output_gate_preactivations[step_index].arr[0]);

      const input_gate_preactivation_gradient = Matrix.init(1, num_hidden).arr[0];
      const forget_gate_preactivation_gradient = Matrix.init(1, num_hidden).arr[0];
      const candidate_gate_preactivation_gradient = Matrix.init(1, num_hidden).arr[0];
      const output_gate_preactivation_gradient = Matrix.init(1, num_hidden).arr[0];
      const cell_gradients = [] as number[];

      for (let unit = 0; unit < num_hidden; unit++) {
        const tanh_cell_state = Math.tanh(cell_state[unit]);
        // dL/dh_t: error from the loss at this step plus the recurrence
        // contribution propagated back from step t + 1
        const hidden_gradient = error_step[unit] + hidden_carry[unit];

        // dL/dz per gate, walking the chain rule back through the gates
        // h = o * tanh(c): the output gate sees tanh(c), the cell state sees
        // o * (1 - tanh(c)^2)
        const output_gate_gradient = hidden_gradient * tanh_cell_state;
        const cell_gradient =
          hidden_gradient * output_gate[unit] * (1 - tanh_cell_state * tanh_cell_state) + cell_carry[unit];
        const input_gate_gradient = cell_gradient * candidate_gate[unit];
        const forget_gate_gradient = cell_gradient * (previous_cell_state ? previous_cell_state[unit] : 0);
        const candidate_gate_gradient = cell_gradient * input_gate[unit];

        // chain rule through each gate's own activation
        input_gate_preactivation_gradient[unit] = input_gate_gradient * input_gate[unit] * (1 - input_gate[unit]);
        forget_gate_preactivation_gradient[unit] = forget_gate_gradient * forget_gate[unit] * (1 - forget_gate[unit]);
        candidate_gate_preactivation_gradient[unit] =
          candidate_gate_gradient * (1 - candidate_gate[unit] * candidate_gate[unit]);
        output_gate_preactivation_gradient[unit] = output_gate_gradient * output_gate[unit] * (1 - output_gate[unit]);

        cell_gradients[unit] = cell_gradient;
      }

      /**
       * dL/dW per gate = [x_t; h_{t-1}; 1] (gate_preactivation_gradient)ᵀ,
       * accumulated over the sequence
       */
      this.accumulate_gate_updates(
        input_gate_updates,
        input_gate_preactivation_gradient,
        input_row,
        previous_hidden_state,
      );
      this.accumulate_gate_updates(
        forget_gate_updates,
        forget_gate_preactivation_gradient,
        input_row,
        previous_hidden_state,
      );
      this.accumulate_gate_updates(
        candidate_gate_updates,
        candidate_gate_preactivation_gradient,
        input_row,
        previous_hidden_state,
      );
      this.accumulate_gate_updates(
        output_gate_updates,
        output_gate_preactivation_gradient,
        input_row,
        previous_hidden_state,
      );

      // dL/dA(t-1): every input/hidden unit connects to all four gate
      // pre-activations, so each receives a contribution from every gate
      const input_gradients = Matrix.init(1, num_input);
      for (let unit = 0; unit < num_input; unit++)
        input_gradients.arr[0][unit] = this.gradient_through_gates(
          unit,
          input_gate_preactivation_gradient,
          forget_gate_preactivation_gradient,
          candidate_gate_preactivation_gradient,
          output_gate_preactivation_gradient,
        );

      const hidden_state_gradients = Matrix.init(1, num_hidden);
      for (let unit = 0; unit < num_hidden; unit++)
        hidden_state_gradients.arr[0][unit] = this.gradient_through_gates(
          num_input + unit,
          input_gate_preactivation_gradient,
          forget_gate_preactivation_gradient,
          candidate_gate_preactivation_gradient,
          output_gate_preactivation_gradient,
        );

      hidden_carry = hidden_state_gradients.arr[0];
      // c_t feeds c_{t-1} scaled by the forget gate of this step
      cell_carry = cell_gradients.map((gradient, unit) => gradient * forget_gate[unit]);
      new_error_steps[step_index] = input_gradients;
    }

    return {
      new_error: new Tensor3D(new_error_steps),
      patch: new LSTMLayerPatch(input_gate_updates, forget_gate_updates, candidate_gate_updates, output_gate_updates),
    };
  }

  // outer product of the shared row [x_t; h_{t-1}; 1] with a single gate's
  // pre-activation gradient row, accumulated into that gate's update matrix
  private accumulate_gate_updates(
    gate_updates: Matrix,
    gate_preactivation_gradient: number[],
    input_row: number[],
    previous_hidden_state: number[] | null,
  ) {
    const num_input = this.input_size;
    const num_hidden = this.hidden_size;

    for (let unit = 0; unit < num_hidden; unit++) {
      const gate_gradient = gate_preactivation_gradient[unit];

      for (let input_index = 0; input_index < num_input; input_index++)
        gate_updates.arr[input_index][unit] += input_row[input_index] * gate_gradient;
      for (let hidden_index = 0; hidden_index < num_hidden; hidden_index++)
        gate_updates.arr[num_input + hidden_index][unit] +=
          (previous_hidden_state ? previous_hidden_state[hidden_index] : 0) * gate_gradient;
      gate_updates.arr[num_input + num_hidden][unit] += gate_gradient;
    }
  }

  // the gradient flowing to one row of the weight block, summed over the four
  // gates: each gate contributes (gate_preactivation_gradient · gate_row)
  private gradient_through_gates(
    weight_row_index: number,
    input_gate_preactivation_gradient: number[],
    forget_gate_preactivation_gradient: number[],
    candidate_gate_preactivation_gradient: number[],
    output_gate_preactivation_gradient: number[],
  ) {
    const num_hidden = this.hidden_size;
    let gradient = 0;
    for (let unit = 0; unit < num_hidden; unit++) {
      gradient += input_gate_preactivation_gradient[unit] * this.input_gate_weights.arr[weight_row_index][unit];
      gradient += forget_gate_preactivation_gradient[unit] * this.forget_gate_weights.arr[weight_row_index][unit];
      gradient += candidate_gate_preactivation_gradient[unit] * this.candidate_gate_weights.arr[weight_row_index][unit];
      gradient += output_gate_preactivation_gradient[unit] * this.output_gate_weights.arr[weight_row_index][unit];
    }
    return gradient;
  }

  to_json(): LSTMLayerSerialized {
    return {
      input_gate_weights: this.input_gate_weights,
      forget_gate_weights: this.forget_gate_weights,
      candidate_gate_weights: this.candidate_gate_weights,
      output_gate_weights: this.output_gate_weights,
      input_size: this.input_size,
      hidden_size: this.hidden_size,
    };
  }

  describe(c: Description) {
    if (c.channels == null) throw new Error("LSTMLayer expects a sequence (channels = time steps)");
    if (c.rows !== 1) throw new Error(`LSTMLayer expects one row per time step, got ${c.rows}`);
    if (c.columns !== this.input_size)
      throw new Error(`LSTMLayer has mismatch between input ${this.input_size} and ${c.columns}`);

    const str = `LSTMLayer (${c.channels} steps, ${this.input_size} -> ${this.hidden_size})`;
    const next: Description = { rows: 1, columns: this.hidden_size, channels: c.channels };
    return { str, next };
  }
}

export class LSTMLayerOutput extends LayerPropogationOutput {
  constructor(
    public readonly input_gate_preactivations: Matrix[], // 1 x hidden per step
    public readonly forget_gate_preactivations: Matrix[], // 1 x hidden per step
    public readonly candidate_gate_preactivations: Matrix[], // 1 x hidden per step
    public readonly output_gate_preactivations: Matrix[], // 1 x hidden per step
    public readonly hidden_states: Matrix[], // 1 x hidden per step (the output)
    public readonly cell_states: Matrix[], // 1 x hidden per step
    public readonly input_steps: Matrix[],
  ) {
    super();
  }

  get_output(): Tensor {
    return new Tensor3D(this.hidden_states);
  }
}

export class LSTMLayerPatch extends Patch {
  constructor(
    public readonly input_gate_updates: Matrix,
    public readonly forget_gate_updates: Matrix,
    public readonly candidate_gate_updates: Matrix,
    public readonly output_gate_updates: Matrix,
  ) {
    super();
  }

  add(other: Patch): Patch {
    if (other instanceof LSTMLayerPatch == false) throw new Error("Not LSTMLayerPatch");
    return new LSTMLayerPatch(
      this.input_gate_updates.add(other.input_gate_updates),
      this.forget_gate_updates.add(other.forget_gate_updates),
      this.candidate_gate_updates.add(other.candidate_gate_updates),
      this.output_gate_updates.add(other.output_gate_updates),
    );
  }

  scale(factor: number) {
    return new LSTMLayerPatch(
      this.input_gate_updates.multiply_scalar(factor),
      this.forget_gate_updates.multiply_scalar(factor),
      this.candidate_gate_updates.multiply_scalar(factor),
      this.output_gate_updates.multiply_scalar(factor),
    );
  }
}

export interface LSTMLayerSerialized {
  input_gate_weights: Matrix;
  forget_gate_weights: Matrix;
  candidate_gate_weights: Matrix;
  output_gate_weights: Matrix;
  input_size: number;
  hidden_size: number;
}
