import { Matrix } from "./matrix";
import fs from "node:fs";

export type Sample = { input: Tensor; label: Tensor };

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
};

export class NeuralNetwork {
  constructor(private readonly loss_fn: Benchmarker) {}

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
    let error: Tensor = this.loss_fn.compute_error(label, a.final);
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
        batch_loss += this.loss_fn.compute_loss(sample.label, pred.final);

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

  train(samples: Sample[], config?: TrainingConfig) {
    const max_epochs = config?.max_epochs ?? 100;
    let last_overall_pct = null as number | null;

    for (let i = 0; i <= max_epochs; i++) {
      const t = this.run_epoch(samples, config);
      console.log(`Epoch ${i}. Loss: ${t}`);

      if (i % 10 == 0) {
        const figures = this.loss_fn.benchmark(this);
        if (last_overall_pct == null || last_overall_pct <= figures) last_overall_pct = figures;
        // else break;
      }
    }

    return this;
  }
}

export abstract class Benchmarker {
  abstract benchmark(nn: NeuralNetwork): number;
  abstract compute_loss(label: Tensor, pred: Tensor): number;
  abstract compute_error(label: Tensor, pred: Tensor): Tensor;
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
