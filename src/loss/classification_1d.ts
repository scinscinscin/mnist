import { Benchmarker, NeuralNetwork, Sample, Tensor, Tensor2D } from "../network";

export class ClassificationCrossEntropy1D extends Benchmarker {
  constructor(
    public readonly samples: Sample[],
    public readonly labal_count: number,
    public readonly top_k: number | null = null,
  ) {
    super();
  }

  compute_error(label: Tensor, pred: Tensor): Tensor2D {
    if (label instanceof Tensor2D == false || pred instanceof Tensor2D == false)
      throw new Error("CrossEntropyLoss1D expects Tensor2D parameters");
    return new Tensor2D(label.matrix.add(pred.matrix.multiply_scalar(-1)));
  }

  compute_loss(label: Tensor, pred: Tensor): number {
    if (label instanceof Tensor2D == false || pred instanceof Tensor2D == false)
      throw new Error("CrossEntropyLoss1D expects Tensor2D parameters");

    const f = pred.matrix;
    let loss = 0;
    for (let k = 0; k < f.arr[0].length; k++) if (label.matrix.arr[0][k] === 1) loss = -Math.log(f.arr[0][k]);
    return loss;
  }

  benchmark(nn: NeuralNetwork) {
    const correct_per_label = new Array(this.labal_count).fill(0);
    const total_per_label = new Array(this.labal_count).fill(0);
    let total_correct = 0;
    let top_k_correct = 0;

    for (const sample of this.samples) {
      const forward = nn.forward(sample.input);
      const output = (forward.final as Tensor2D).matrix.arr[0];
      const predicted = output.indexOf(Math.max(...output));
      const true_label = (sample.label as Tensor2D).matrix.arr[0].indexOf(1);

      total_per_label[true_label]++;
      if (predicted === true_label) {
        correct_per_label[true_label]++;
        total_correct++;
      }

      if (this.top_k != null) {
        const top_indices = output
          .map((score, idx) => ({ score, idx }))
          .sort((a, b) => b.score - a.score)
          .slice(0, this.top_k)
          .map((e) => e.idx);
        if (top_indices.includes(true_label)) top_k_correct++;
      }
    }

    console.log("\nBenchmark results:");
    for (let i = 0; i < this.labal_count; i++) {
      const pct = total_per_label[i] > 0 ? ((correct_per_label[i] / total_per_label[i]) * 100).toFixed(1) : "0.0";
      console.log(`Label ${i}: ${correct_per_label[i]}/${total_per_label[i]} correct (${pct}%)`);
    }

    const overall_pct_n = (total_correct / this.samples.length) * 100;
    const overall_pct = overall_pct_n.toFixed(1);
    console.log(`Overall accuracy: ${total_correct}/${this.samples.length} correct (${overall_pct}%)`);

    if (this.top_k != null) {
      const top_k_pct_n = (top_k_correct / this.samples.length) * 100;
      const top_k_pct = top_k_pct_n.toFixed(1);
      console.log(`Top-k(k=${this.top_k}) accuracy: ${top_k_correct}/${this.samples.length} correct (${top_k_pct}%)`);
    }

    console.log();
    return overall_pct_n;
  }
}
