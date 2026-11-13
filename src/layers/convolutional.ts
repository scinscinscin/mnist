import { Activation1D } from "../activations";
import { Matrix, mulberry32 } from "../matrix";

export class ConvolutionalLayer {
  kernels: Matrix[][];
  bias: number[];

  constructor(
    kconfig: { rows: number; cols: number; input_channels: number; output_channels: number },
    public readonly activation: Activation1D,
    seed: number,
    public readonly options?: { padding?: number; stride?: number; dilation?: number },
  ) {
    const kernels = [] as Matrix[][];
    const bias = [] as number[];
    const rng = mulberry32(seed++);

    // initialize kernels weights and biases
    for (let i = 0; i < kconfig.output_channels; i++) {
      const kernel = [] as Matrix[];
      for (let j = 0; j < kconfig.input_channels; j++)
        kernel.push(Matrix.from_random(kconfig.rows, kconfig.cols, seed++));

      kernels.push(kernel);
      bias.push(rng());
    }

    this.kernels = kernels;
    this.bias = bias;
  }

  forward(input_channels: Matrix[]) {
    // make sure that all 3 channels in the input have the same size
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

    return { raw: output_channels_raw, output: output_channels };
  }
}

export class MaxPoolingLayer {
  rows: number;
  cols: number;
  stride: number;

  constructor(kernel_size: { rows: number; cols: number; stride?: number }) {
    this.rows = kernel_size.rows;
    this.cols = kernel_size.cols;
    // 0 means default: stride equals the kernel size (non-overlapping)
    this.stride = kernel_size.stride ?? 0;
  }

  forward(input_channels: Matrix[]) {
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

    return input_channels.map((input) => {
      const pooled = Matrix.init(output_rows, output_cols);

      for (let i = 0; i < output_rows; i++)
        for (let j = 0; j < output_cols; j++) {
          let max = -Infinity;

          for (let r = 0; r < this.rows; r++)
            for (let c = 0; c < this.cols; c++)
              max = Math.max(max, input.arr[i * stride_rows + r][j * stride_cols + c]);

          pooled.arr[i][j] = max;
        }

      return pooled;
    });
  }
}

export class AveragePoolingLayer {
  rows: number;
  cols: number;
  stride: number;

  constructor(kernel_size: { rows: number; cols: number; stride?: number }) {
    this.rows = kernel_size.rows;
    this.cols = kernel_size.cols;
    // 0 means default: stride equals the kernel size (non-overlapping)
    this.stride = kernel_size.stride ?? 0;
  }

  forward(input_channels: Matrix[]): Matrix[] {
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

    return input_channels.map((input) => {
      const pooled = Matrix.init(output_rows, output_cols);

      for (let i = 0; i < output_rows; i++)
        for (let j = 0; j < output_cols; j++) {
          let cumulative = 0;
          for (let r = 0; r < this.rows; r++)
            for (let c = 0; c < this.cols; c++) cumulative += input.arr[i * stride_rows + r][j * stride_cols + c];
          pooled.arr[i][j] = cumulative / (this.rows * this.cols);
        }

      return pooled;
    });
  }
}

export class FlattenLayer {
  forward(input_channels: Matrix[]): Matrix {
    const flat = [] as number[];
    for (const channel of input_channels) for (const row of channel.arr) flat.push(...row);
    return Matrix.from_existing([flat]);
  }
}
