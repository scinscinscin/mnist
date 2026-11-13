import { Activation1D } from "./activations";
import { Matrix, mulberry32 } from "./matrix";
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
