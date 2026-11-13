import { Sample } from "./load";
import { Matrix } from "./matrix";
import fs from "node:fs";

export abstract class Tensor {
  t = "";
}

export class Tensor2D extends Tensor {
  constructor(public matrix: Matrix) {
    super();
  }
}

export class Tensor3D extends Tensor {
  constructor(public matrix: Matrix[]) {
    super();
  }
}

export type TrainingConfig = {
  max_epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  benchmark: Benchmarker;
};

export class NeuralNetwork {
  layers: Layer[] = [];
  add_layer(layer: Layer) {
    this.layers.push(layer);
    return this;
  }

  forward(input: Tensor) {
    let i = input;
    let intermediates = [] as LayerPropogationOutput[];

    for (const layer of this.layers) {
      const x = layer.forward(i);
      i = x.get_output();
      intermediates.push(x);
    }

    return { final: i, intermediates };
  }

  get_updates(label: Tensor, a: ReturnType<NeuralNetwork["forward"]>) {
    if (label instanceof Tensor2D == false || a.final instanceof Tensor2D == false) {
      throw new Error("Only Tensor2D supported at the end for now");
    }

    let error: Tensor = new Tensor2D(label.matrix.add(a.final.matrix.multiply_scalar(-1)));
    const layer_updates: Patch[] = [];

    for (let layer_idx = this.layers.length - 1; layer_idx >= 0; layer_idx--) {
      const layer = this.layers[layer_idx];
      const propogation_output = a.intermediates[layer_idx];
      const { patch, new_error } = layer.get_updates(error, propogation_output);

      layer_updates.push(patch);
      error = new_error;
    }

    return layer_updates;
  }

  save(file: string) {
    const c = this.layers.map((c) => c.to_json());
    const o = { weights: c };
    fs.writeFileSync(file, JSON.stringify(o));
  }

  describe(c: Description) {
    let current = c;

    console.log("Neural Network:");
    console.log(` - Input dimensions: (${current.rows} x ${current.columns} x ${current.channels ?? 1})`);
    for (const layer of this.layers) {
      const { next, str } = layer.describe(current);
      console.log(` - ${str}`);
      current = next;
    }
    console.log(` - Output dimensions: (${current.rows} x ${current.columns} x ${current.channels ?? 1})`);
  }

  private run_epoch(samples: Sample[], config?: TrainingConfig) {
    const batch_size = config?.batch_size ?? 32;
    const learning_rate = config?.learning_rate ?? 1.0;

    let total_loss = 0;
    let count = 0;

    for (let b = 0; b < samples.length; b += batch_size) {
      const chunk = samples.slice(b, b + batch_size);
      let cumulative_patches: Patch[] | null = null;
      let batch_loss = 0;

      for (const sample of chunk) {
        const pred = this.forward(sample.input);

        // cross-entropy: -log(probability assigned to the true class)
        const f = (pred.final as Tensor2D).matrix;
        let loss = 0;
        for (let k = 0; k < f.arr[0].length; k++)
          if (sample.label.matrix.arr[0][k] === 1) loss = -Math.log(f.arr[0][k]);
        batch_loss += loss;

        const sample_update = this.get_updates(sample.label, pred);
        if (cumulative_patches == null) cumulative_patches = sample_update;
        else cumulative_patches = cumulative_patches.map((patch, k) => patch.add(sample_update[k]));
      }

      // get_updates returns updates in reverse order (last layer first), so reverse
      // before zipping onto this.layers (first layer first).
      const normalized = cumulative_patches!.map((e) => e.scale(learning_rate / chunk.length)).reverse();
      for (let i = 0; i < this.layers.length; i++) this.layers[i].update(normalized[i]);

      total_loss += batch_loss;
      count += chunk.length;
    }

    return total_loss / count;
  }

  train(samples: Sample[], config: TrainingConfig) {
    const max_epochs = config?.max_epochs ?? 100;
    let last_overall_pct = null as number | null;

    for (let i = 0; i < max_epochs; i++) {
      const t = this.run_epoch(samples, config);
      console.log(`Epoch ${i}. Loss: ${t}`);

      if (i % 10 == 0) {
        const figures = config.benchmark.benchmark(this);
        if (last_overall_pct == null || last_overall_pct <= figures) last_overall_pct = figures;
        else break;
      }
    }

    return this;
  }
}

export abstract class Benchmarker {
  abstract benchmark(nn: NeuralNetwork): number;
}

export class ClassificationCrossEntropyBenchmark extends Benchmarker {
  constructor(
    public readonly samples: Sample[],
    public readonly labal_count: number,
  ) {
    super();
  }

  benchmark(nn: NeuralNetwork) {
    const correct_per_label = new Array(this.labal_count).fill(0);
    const total_per_label = new Array(this.labal_count).fill(0);
    let total_correct = 0;

    for (const sample of this.samples) {
      const forward = nn.forward(sample.input);
      const output = (forward.final as Tensor2D).matrix.arr[0];
      const predicted = output.indexOf(Math.max(...output));
      const true_label = sample.label.matrix.arr[0].indexOf(1);

      total_per_label[true_label]++;
      if (predicted === true_label) {
        correct_per_label[true_label]++;
        total_correct++;
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
    return overall_pct_n;
  }
}

export type Description = { rows: number; columns: number; channels?: number };

export abstract class Layer<JSONFormat = any> {
  abstract forward(input: Tensor): LayerPropogationOutput;
  abstract get_updates(tensor: Tensor, activations: LayerPropogationOutput): { new_error: Tensor; patch: Patch };
  abstract update(patch: Patch): void;

  abstract to_json(): JSONFormat;
  abstract describe(c: Description): { next: Description; str: string };
}

export abstract class LayerPropogationOutput {
  abstract get_output(): Tensor;
}

export abstract class Patch {
  abstract add(p: Patch): Patch;
  abstract scale(c: number): Patch;
}

export class NoopPatch extends Patch {
  add() {
    return this;
  }

  scale() {
    return this;
  }
}
