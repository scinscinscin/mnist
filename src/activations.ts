export type Activation1D = {
  forward: (x: number[]) => number[];
  backward: (x: number) => number;
  id: string;
};

const _sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
export const sigmoid: Activation1D = {
  id: "sigmoid_1d",
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
export const softmax: Activation1D = {
  id: "softmax_1d",
  forward: function softmax_row(row: number[]): number[] {
    const max = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  },
  backward: () => 1,
};
