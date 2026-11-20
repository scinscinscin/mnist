import { Matrix } from "../matrix";
import { Benchmarker, NeuralNetwork, Sample, Tensor, Tensor3D } from "../network";

export class ClassificationCrossEntropy2D extends Benchmarker {
  constructor(
    public readonly samples: Sample[],
    public readonly labal_count: number,
    public readonly top_k: number | null = null,
  ) {
    super();
  }

  compute_error(label: Tensor, pred: Tensor): Tensor3D {
    if (label instanceof Tensor3D == false || pred instanceof Tensor3D == false)
      throw new Error("CrossEntropyLoss2D expects Tensor3D parameters");

    const result = [] as Matrix[];
    for (let i = 0; i < pred.matrix.length; i++) result[i] = label.matrix[i].add(pred.matrix[i].multiply_scalar(-1));
    return new Tensor3D(result);
  }

  compute_loss(label: Tensor, pred: Tensor): number {
    if (label instanceof Tensor3D == false || pred instanceof Tensor3D == false)
      throw new Error("CrossEntropyLoss2D expects Tensor3D parameters");

    let loss = 0;
    for (let i = 0; i < pred.matrix.length; i++) {
      const f = pred.matrix[i];
      const l = label.matrix[i];
      for (let k = 0; k < f.arr[0].length; k++) if (l.arr[0][k] === 1) loss -= Math.log(f.arr[0][k]);
    }
    return loss;
  }

  benchmark(nn: NeuralNetwork) {
    const correct_per_label = new Array(this.labal_count).fill(0);
    const total_per_label = new Array(this.labal_count).fill(0);
    let total_correct = 0;
    let total_count = 0;
    let top_k_correct = 0;

    for (const sample of this.samples) {
      const forward = nn.forward(sample.input);
      const output = (forward.final as Tensor3D).matrix;
      const labels = (sample.label as Tensor3D).matrix;

      for (let i = 0; i < output.length; i++) {
        const row = output[i].arr[0];
        const predicted = row.indexOf(Math.max(...row));
        const true_label = labels[i].arr[0].indexOf(1);

        total_per_label[true_label]++;
        total_count++;
        if (predicted === true_label) {
          correct_per_label[true_label]++;
          total_correct++;
        }

        if (this.top_k != null) {
          const top_indices = row
            .map((score, idx) => ({ score, idx }))
            .sort((a, b) => b.score - a.score)
            .slice(0, this.top_k)
            .map((e) => e.idx);
          if (top_indices.includes(true_label)) top_k_correct++;
        }
      }
    }

    console.log("\nBenchmark results:");
    for (let i = 0; i < this.labal_count; i++) {
      const pct = total_per_label[i] > 0 ? ((correct_per_label[i] / total_per_label[i]) * 100).toFixed(1) : "0.0";
      console.log(`Label ${i}: ${correct_per_label[i]}/${total_per_label[i]} correct (${pct}%)`);
    }

    const overall_pct_n = (total_correct / total_count) * 100;
    const overall_pct = overall_pct_n.toFixed(1);
    console.log(`Overall accuracy: ${total_correct}/${total_count} correct (${overall_pct}%)`);

    if (this.top_k != null) {
      const top_k_pct_n = (top_k_correct / total_count) * 100;
      const top_k_pct = top_k_pct_n.toFixed(1);
      console.log(`Top-k(k=${this.top_k}) accuracy: ${top_k_correct}/${total_count} correct (${top_k_pct}%)`);
    }

    console.log();
    return overall_pct_n;
  }
}
