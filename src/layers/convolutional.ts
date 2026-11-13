import { Activation1D } from "../activations";
import { Matrix } from "../matrix";
import { Description, Layer, LayerPropogationOutput, NoopPatch, Patch, Tensor, Tensor2D, Tensor3D } from "../network";

interface ConvolutionalLayerSerialized {
  kernels: Matrix[][];
  bias: number[];
  activation: string;
}

export class ConvolutionalLayer extends Layer<ConvolutionalLayerSerialized> {
  kernels: Matrix[][];
  bias: number[];

  constructor(
    kconfig: { rows: number; cols: number; input_channels: number; output_channels: number },
    public readonly activation: Activation1D,
    seed: number,
    public readonly options?: { padding?: number; stride?: number; dilation?: number },
  ) {
    super();
    const kernels = [] as Matrix[][];
    const bias = [] as number[];

    // He initialization: zero-centered weights scaled by the fan-in keep
    // pre-activation magnitudes O(1) through the conv stack, so the ReLU and
    // downstream activation units stay out of saturation and gradients flow
    const scale = Math.sqrt(2 / (kconfig.rows * kconfig.cols * kconfig.input_channels));

    // initialize kernels weights and biases
    for (let i = 0; i < kconfig.output_channels; i++) {
      const kernel = [] as Matrix[];
      for (let j = 0; j < kconfig.input_channels; j++)
        kernel.push(Matrix.from_uniform(kconfig.rows, kconfig.cols, seed++, scale));
      kernels.push(kernel);
      bias.push(0);
    }

    this.kernels = kernels;
    this.bias = bias;
  }

  forward(input: Tensor): ConvolutionalLayerOutput {
    if (input instanceof Tensor3D == false) throw new Error("Input not suitable for ConvolutionalLayer");

    const input_channels = input.matrix;
    // make sure that all channels in the input have the same size
    const { rows: input_rows, cols: input_cols } = input_channels[0];
    const matches = input_channels.every((m) => m.rows == input_rows && m.cols == input_cols);

    if (!matches) throw new Error("Input tensor matrices do not have consistent row and col count");
    if (input_channels.length !== this.kernels[0].length)
      throw new Error(`expected ${this.kernels[0].length} input channels, got ${input_channels.length}`);

    // 0 means default: stride 1 and no dilation (padding 0 = no padding)
    const { stride: _stride = 0, dilation: _dilation = 0, padding = 0 } = this.options ?? {};
    const stride = _stride || 1;
    const dilation = _dilation || 1;

    const { rows: kernel_rows, cols: kernel_cols } = this.kernels[0][0];
    const effective_kernel_rows = dilation * (kernel_rows - 1) + 1;
    const effective_kernel_cols = dilation * (kernel_cols - 1) + 1;
    const output_rows = Math.floor((input_rows + 2 * padding - effective_kernel_rows) / stride) + 1;
    const output_cols = Math.floor((input_cols + 2 * padding - effective_kernel_cols) / stride) + 1;

    const output_channels = [] as Matrix[];
    const output_channels_raw = [] as Matrix[];

    for (const kernel of this.kernels) {
      // accumulate the pre-activation value map: sum over the cross-correlations
      // of each input channel with its corresponding kernel
      let raw = Matrix.init(output_rows, output_cols);

      for (let ci = 0; ci < kernel.length; ci++) {
        const kernel_slice = kernel[ci];
        const input_slice = input_channels[ci];

        // Apply the convolutions. The kernel is walked over the input with the
        // given stride and dilation; positions covered by the zero padding fall
        // outside the input and contribute nothing.
        for (let i = 0; i < output_rows; i++)
          for (let j = 0; j < output_cols; j++)
            for (let r = 0; r < kernel_rows; r++)
              for (let c = 0; c < kernel_cols; c++) {
                const in_r = i * stride - padding + r * dilation;
                const in_c = j * stride - padding + c * dilation;
                if (in_r >= 0 && in_r < input_rows && in_c >= 0 && in_c < input_cols)
                  raw.arr[i][j] += kernel_slice.arr[r][c] * input_slice.arr[in_r][in_c];
              }
      }

      // add the bias, then apply the activation row-wise
      raw = raw.map((x) => x + this.bias[output_channels.length]);
      output_channels_raw.push(raw);
      output_channels.push(Matrix.from_existing(raw.arr.map((row) => this.activation.forward(row))));
    }

    return new ConvolutionalLayerOutput(output_channels_raw, output_channels, input_channels);
  }

  update(patch: Patch) {
    if (patch instanceof ConvolutionalLayerPatch == false) throw new Error("Incorrect class");

    this.kernels = this.kernels.map((kernel, oc) => kernel.map((k, ci) => k.add(patch.updates[oc][ci])));
    this.bias = this.bias.map((b, i) => b + patch.bias[i]);
  }

  get_updates(error: Tensor, propogation_output: LayerPropogationOutput) {
    if (error instanceof Tensor3D == false || propogation_output instanceof ConvolutionalLayerOutput == false)
      throw new Error("Incorrect classes passed to get_updates");

    const { raw, input } = propogation_output;
    const error_channels = error.matrix;
    if (error_channels.length !== this.kernels.length)
      throw new Error(`expected ${this.kernels.length} error channels, got ${error_channels.length}`);

    const { rows: kernel_rows, cols: kernel_cols } = this.kernels[0][0];
    const { rows: input_rows, cols: input_cols } = input[0];
    const { stride: _stride = 0, dilation: _dilation = 0, padding = 0 } = this.options ?? {};
    const stride = _stride || 1;
    const dilation = _dilation || 1;

    const updates = [] as Matrix[][];
    const bias_updates = [] as number[];
    const new_error = input.map((m) => Matrix.init(m.rows, m.cols));

    for (let oc = 0; oc < this.kernels.length; oc++) {
      const kernel = this.kernels[oc];
      const kernel_updates = kernel.map((k) => Matrix.init(k.rows, k.cols));
      let bias_update = 0;

      for (let ci = 0; ci < kernel.length; ci++) {
        const input_slice = input[ci];

        for (let i = 0; i < error_channels[oc].rows; i++)
          for (let j = 0; j < error_channels[oc].cols; j++) {
            // Compute common factor dA/dZ * dC/dA used in both weight and loss adjustment
            const common = this.activation.backward(raw[oc].arr[i][j]) * error_channels[oc].arr[i][j];
            bias_update += common;

            /**
             * dC/dW = input * dA/dZ * dC/dA
             * dC/dA_in = kernel * dA/dZ * dC/dA, scattered back to the input positions
             */
            for (let r = 0; r < kernel_rows; r++)
              for (let c = 0; c < kernel_cols; c++) {
                const in_r = i * stride - padding + r * dilation;
                const in_c = j * stride - padding + c * dilation;
                if (in_r >= 0 && in_r < input_rows && in_c >= 0 && in_c < input_cols) {
                  kernel_updates[ci].arr[r][c] += input_slice.arr[in_r][in_c] * common;
                  new_error[ci].arr[in_r][in_c] += kernel[ci].arr[r][c] * common;
                }
              }
          }
      }

      updates.push(kernel_updates);
      bias_updates.push(bias_update);
    }

    return {
      new_error: new Tensor3D(new_error),
      patch: new ConvolutionalLayerPatch(updates, bias_updates),
    };
  }

  to_json(): ConvolutionalLayerSerialized {
    return { kernels: this.kernels, bias: this.bias, activation: this.activation.id };
  }

  describe(c: Description) {
    const channels = c.channels ?? 1;
    if (channels !== this.kernels[0].length)
      throw new Error(
        `ConvolutionalLayer has mismatch between input channels ${this.kernels[0].length} and ${channels}`,
      );

    const { rows: kernel_rows, cols: kernel_cols } = this.kernels[0][0];
    const { stride: _stride = 0, dilation: _dilation = 0, padding = 0 } = this.options ?? {};
    const stride = _stride || 1;
    const dilation = _dilation || 1;
    const effective_kernel_rows = dilation * (kernel_rows - 1) + 1;
    const effective_kernel_cols = dilation * (kernel_cols - 1) + 1;
    const output_rows = Math.floor((c.rows + 2 * padding - effective_kernel_rows) / stride) + 1;
    const output_cols = Math.floor((c.columns + 2 * padding - effective_kernel_cols) / stride) + 1;

    return {
      next: { rows: output_rows, columns: output_cols, channels: this.kernels.length },
      str: `ConvolutionalLayer (${kernel_rows} x ${kernel_cols}, ${channels} -> ${this.kernels.length}, stride ${stride}, dilation ${dilation}, padding ${padding})`,
    };
  }
}

export class ConvolutionalLayerOutput extends LayerPropogationOutput {
  constructor(
    public readonly raw: Matrix[],
    public readonly output: Matrix[],
    public readonly input: Matrix[],
  ) {
    super();
  }

  get_output(): Tensor {
    return new Tensor3D(this.output);
  }
}

export class ConvolutionalLayerPatch extends Patch {
  constructor(
    public readonly updates: Matrix[][],
    public readonly bias: number[],
  ) {
    super();
  }

  add(p: Patch): Patch {
    if (p instanceof ConvolutionalLayerPatch == false) throw new Error("Not ConvolutionalLayerPatch");

    return new ConvolutionalLayerPatch(
      this.updates.map((kernel, oc) => kernel.map((m, ci) => m.add(p.updates[oc][ci]))),
      this.bias.map((b, i) => b + p.bias[i]),
    );
  }

  scale(c: number): Patch {
    return new ConvolutionalLayerPatch(
      this.updates.map((kernel) => kernel.map((m) => m.multiply_scalar(c))),
      this.bias.map((b) => b * c),
    );
  }
}

interface MaxPoolingLayerSerialized {
  rows: number;
  cols: number;
  stride: number;
}

export class MaxPoolingLayer extends Layer<MaxPoolingLayerSerialized> {
  rows: number;
  cols: number;
  stride: number;

  constructor(kernel_size: { rows: number; cols: number; stride?: number }) {
    super();
    this.rows = kernel_size.rows;
    this.cols = kernel_size.cols;
    // 0 means default: stride equals the kernel size (non-overlapping)
    this.stride = kernel_size.stride ?? 0;
  }

  forward(input: Tensor): MaxPoolingLayerOutput {
    if (input instanceof Tensor3D == false) throw new Error("Input not suitable for MaxPoolingLayer");

    const input_channels = input.matrix;
    // make sure that all channels in the input have the same size
    const { rows: input_rows, cols: input_cols } = input_channels[0];
    const matches = input_channels.every((m) => m.rows == input_rows && m.cols == input_cols);

    if (!matches) throw new Error("Input tensor matrices do not have consistent row and col count");
    if (this.stride == 0 && (input_rows % this.rows != 0 || input_cols % this.cols != 0))
      throw new Error(`Pooling layer of size ${this.rows}x${this.cols} cannot divide ${input_rows}x${input_cols}`);

    // 0 means default: stride equals the kernel size (non-overlapping)
    const stride_rows = this.stride || this.rows;
    const stride_cols = this.stride || this.cols;

    // the feature map shrinks by the kernel size, walking with the given stride
    const output_rows = Math.floor((input_rows - this.rows) / stride_rows) + 1;
    const output_cols = Math.floor((input_cols - this.cols) / stride_cols) + 1;

    const output = [] as Matrix[];
    const mask = [] as Matrix[];

    for (const input_channel of input_channels) {
      const pooled = Matrix.init(output_rows, output_cols);
      // marks the argmax of each window; only that element receives the error
      const mask_channel = Matrix.init(input_channel.rows, input_channel.cols);

      for (let i = 0; i < output_rows; i++)
        for (let j = 0; j < output_cols; j++) {
          let max = -Infinity;
          let max_r = 0;
          let max_c = 0;

          for (let r = 0; r < this.rows; r++)
            for (let c = 0; c < this.cols; c++) {
              const v = input_channel.arr[i * stride_rows + r][j * stride_cols + c];
              if (v > max) {
                max = v;
                max_r = r;
                max_c = c;
              }
            }

          pooled.arr[i][j] = max;
          mask_channel.arr[i * stride_rows + max_r][j * stride_cols + max_c] = 1;
        }

      output.push(pooled);
      mask.push(mask_channel);
    }

    return new MaxPoolingLayerOutput(output, mask);
  }

  update() {}

  get_updates(error: Tensor, propogation_output: LayerPropogationOutput) {
    if (error instanceof Tensor3D == false || propogation_output instanceof MaxPoolingLayerOutput == false)
      throw new Error("Incorrect classes passed to get_updates");

    const { mask } = propogation_output;
    if (error.matrix.length !== mask.length) throw new Error("Pooling layer error channel count mismatch");

    const stride_rows = this.stride || this.rows;
    const stride_cols = this.stride || this.cols;

    // scatter each error value back to the argmax of its window
    const new_error = error.matrix.map((e, ch) => {
      const input_rows = (e.rows - 1) * stride_rows + this.rows;
      const input_cols = (e.cols - 1) * stride_cols + this.cols;
      const unrolled = Matrix.init(input_rows, input_cols);

      for (let i = 0; i < e.rows; i++)
        for (let j = 0; j < e.cols; j++)
          for (let r = 0; r < this.rows; r++)
            for (let c = 0; c < this.cols; c++)
              unrolled.arr[i * stride_rows + r][j * stride_cols + c] =
                e.arr[i][j] * mask[ch].arr[i * stride_rows + r][j * stride_cols + c];

      return unrolled;
    });

    return { new_error: new Tensor3D(new_error), patch: new NoopPatch() };
  }

  to_json(): MaxPoolingLayerSerialized {
    return { rows: this.rows, cols: this.cols, stride: this.stride };
  }

  describe(c: Description) {
    const stride_rows = this.stride || this.rows;
    const stride_cols = this.stride || this.cols;
    const output_rows = Math.floor((c.rows - this.rows) / stride_rows) + 1;
    const output_cols = Math.floor((c.columns - this.cols) / stride_cols) + 1;
    if (output_rows <= 0 || output_cols <= 0)
      throw new Error(
        `MaxPoolingLayer of size ${this.rows}x${this.cols} is larger than the ${c.rows}x${c.columns} input`,
      );

    return {
      next: { rows: output_rows, columns: output_cols, channels: c.channels ?? 1 },
      str: `MaxPoolingLayer (${this.rows} x ${this.cols}, stride ${this.stride})`,
    };
  }
}

export class MaxPoolingLayerOutput extends LayerPropogationOutput {
  constructor(
    public readonly output: Matrix[],
    public readonly mask: Matrix[],
  ) {
    super();
  }

  get_output(): Tensor {
    return new Tensor3D(this.output);
  }
}

interface AveragePoolingLayerSerialized {
  rows: number;
  cols: number;
  stride: number;
}

export class AveragePoolingLayer extends Layer<AveragePoolingLayerSerialized> {
  rows: number;
  cols: number;
  stride: number;

  constructor(kernel_size: { rows: number; cols: number; stride?: number }) {
    super();
    this.rows = kernel_size.rows;
    this.cols = kernel_size.cols;
    // 0 means default: stride equals the kernel size (non-overlapping)
    this.stride = kernel_size.stride ?? 0;
  }

  forward(input: Tensor): AveragePoolingLayerOutput {
    if (input instanceof Tensor3D == false) throw new Error("Input not suitable for AveragePoolingLayer");

    const input_channels = input.matrix;
    // make sure that all channels in the input have the same size
    const { rows: input_rows, cols: input_cols } = input_channels[0];
    const matches = input_channels.every((m) => m.rows == input_rows && m.cols == input_cols);

    if (!matches) throw new Error("Input tensor matrices do not have consistent row and col count");
    if (this.stride == 0 && (input_rows % this.rows != 0 || input_cols % this.cols != 0))
      throw new Error(`Pooling layer of size ${this.rows}x${this.cols} cannot divide ${input_rows}x${input_cols}`);

    // 0 means default: stride equals the kernel size (non-overlapping)
    const stride_rows = this.stride || this.rows;
    const stride_cols = this.stride || this.cols;

    // the feature map shrinks by the kernel size, walking with the given stride
    const output_rows = Math.floor((input_rows - this.rows) / stride_rows) + 1;
    const output_cols = Math.floor((input_cols - this.cols) / stride_cols) + 1;

    return new AveragePoolingLayerOutput(
      input_channels.map((input) => {
        const pooled = Matrix.init(output_rows, output_cols);

        for (let i = 0; i < output_rows; i++)
          for (let j = 0; j < output_cols; j++) {
            let cumulative = 0;
            for (let r = 0; r < this.rows; r++)
              for (let c = 0; c < this.cols; c++) cumulative += input.arr[i * stride_rows + r][j * stride_cols + c];
            pooled.arr[i][j] = cumulative / (this.rows * this.cols);
          }

        return pooled;
      }),
    );
  }

  update() {}

  get_updates(error: Tensor, propogation_output: LayerPropogationOutput) {
    if (error instanceof Tensor3D == false || propogation_output instanceof AveragePoolingLayerOutput == false)
      throw new Error("Incorrect classes passed to get_updates");

    const stride_rows = this.stride || this.rows;
    const stride_cols = this.stride || this.cols;

    // the mean distributes each error value equally across its window
    const new_error = error.matrix.map((e) => {
      const input_rows = (e.rows - 1) * stride_rows + this.rows;
      const input_cols = (e.cols - 1) * stride_cols + this.cols;
      const unrolled = Matrix.init(input_rows, input_cols);

      for (let i = 0; i < e.rows; i++)
        for (let j = 0; j < e.cols; j++)
          for (let r = 0; r < this.rows; r++)
            for (let c = 0; c < this.cols; c++)
              unrolled.arr[i * stride_rows + r][j * stride_cols + c] = e.arr[i][j] / (this.rows * this.cols);

      return unrolled;
    });

    return { new_error: new Tensor3D(new_error), patch: new NoopPatch() };
  }

  to_json(): AveragePoolingLayerSerialized {
    return { rows: this.rows, cols: this.cols, stride: this.stride };
  }

  describe(c: Description) {
    const stride_rows = this.stride || this.rows;
    const stride_cols = this.stride || this.cols;
    const output_rows = Math.floor((c.rows - this.rows) / stride_rows) + 1;
    const output_cols = Math.floor((c.columns - this.cols) / stride_cols) + 1;
    if (output_rows <= 0 || output_cols <= 0)
      throw new Error(
        `AveragePoolingLayer of size ${this.rows}x${this.cols} is larger than the ${c.rows}x${c.columns} input`,
      );

    return {
      next: { rows: output_rows, columns: output_cols, channels: c.channels ?? 1 },
      str: `AveragePoolingLayer (${this.rows} x ${this.cols}, stride ${this.stride})`,
    };
  }
}

export class AveragePoolingLayerOutput extends LayerPropogationOutput {
  constructor(public readonly output: Matrix[]) {
    super();
  }

  get_output(): Tensor {
    return new Tensor3D(this.output);
  }
}

interface FlattenLayerLayerSerialized {
  rows: number;
  cols: number;
  channels: number;
}

export class FlattenLayerLayerOutput extends LayerPropogationOutput {
  constructor(public readonly output: Matrix) {
    super();
  }

  get_output() {
    return new Tensor2D(this.output);
  }
}

export class FlattenLayer extends Layer<FlattenLayerLayerSerialized> {
  constructor(
    public readonly rows: number,
    public readonly cols: number,
    public readonly channels: number = 1,
  ) {
    super();
  }

  forward(input: Tensor): FlattenLayerLayerOutput {
    if (input instanceof Tensor3D == false) throw new Error("Input not suitable for FlattenLayer");

    const input_channels = input.matrix;
    const flat = [] as number[];
    for (const channel of input_channels) for (const row of channel.arr) flat.push(...row);
    return new FlattenLayerLayerOutput(Matrix.from_existing([flat]));
  }

  get_updates(tensor: Tensor, activation: LayerPropogationOutput) {
    if (tensor instanceof Tensor2D == false || activation instanceof FlattenLayerLayerOutput == false)
      throw new Error("Incorrect classes passed to get_updates");

    const error_matrix = tensor.matrix;
    const expected_cols = this.rows * this.cols * this.channels;
    if (error_matrix.rows !== 1 || error_matrix.cols !== expected_cols)
      throw new Error(
        `FlattenLayer error matrix is ${error_matrix.rows}x${error_matrix.cols}, expected 1x${expected_cols}`,
      );

    // unroll the 1d error row back into per-channel feature maps, inverting
    // the forward pass: channels were flattened first, then each row in order
    const flat = error_matrix.arr[0];
    const unrolled = [] as Matrix[];
    let offset = 0;

    for (let ch = 0; ch < this.channels; ch++) {
      const channel = [] as number[][];
      for (let r = 0; r < this.rows; r++)
        channel.push(flat.slice(offset + r * this.cols, offset + (r + 1) * this.cols));

      unrolled.push(Matrix.from_existing(channel));
      offset += this.rows * this.cols;
    }

    return { new_error: new Tensor3D(unrolled), patch: new NoopPatch() };
  }

  update() {}

  to_json() {
    return { rows: this.rows, cols: this.cols, channels: this.channels };
  }

  describe(c: Description) {
    if (c.rows !== this.rows)
      throw new Error(`FlattenLayer row dimension mismatch. Expected ${this.rows}, received ${c.rows}`);
    if (c.columns !== this.cols)
      throw new Error(`FlattenLayer column dimension mismatch. Expected ${this.cols}, received ${c.columns}`);
    if ((c.channels ?? 1) !== this.channels)
      throw new Error(
        `FlattenLayer channel dimension mismatch. Expected ${this.channels}, received ${c.channels ?? 1}`,
      );

    const columns = this.rows * this.cols * this.channels;
    return {
      next: { rows: 1, channels: 1, columns },
      str: `FlattenLayer (${this.rows} x ${this.cols} x ${this.channels})`,
    };
  }
}
