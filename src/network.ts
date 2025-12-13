import { ActivationFunction, Activations } from "./activations";
import { Layer } from "./layer";
import { Matrix } from "./matrix";
import { RNG } from "./rng";

interface NetworkOptions {
  layers: { size: number; activation: ActivationFunction | keyof typeof Activations }[];
  rng: RNG;
  lr: number;
}

export class Network {
  layers: Layer[] = [];
  lr = 0.0000001;

  constructor({ layers, rng, lr }: NetworkOptions) {
    this.lr = lr;

    for (let i = 1; i < layers.length; i++) {
      const _activationFunction = layers[i].activation;
      const activationFunction =
        typeof _activationFunction === "string" ? Activations[_activationFunction] : _activationFunction;

      this.layers.push(
        new Layer({
          input: layers[i - 1].size,
          output: layers[i].size,
          activationFunction,
          rng: () => rng.nextFloat(),
        })
      );
    }
  }

  forward(input: Matrix) {
    let currentActivation = input;
    const intermediateActivations: Matrix[] = [];

    for (let i = 0; i < this.layers.length; i++) {
      const { result, rawActivations } = this.layers[i].forward(currentActivation);
      intermediateActivations.push(rawActivations);
      currentActivation = result;
    }

    return { intermediateActivations, result: currentActivation };
  }

  backward(expectedOutput: Matrix, intermediateActivations: Matrix[]) {
    let difference = expectedOutput.subtractWith(intermediateActivations[intermediateActivations.length - 1]);

    for (let i = intermediateActivations.length - 1; i >= 1; i--) {
      const currentLayer = this.layers[i - 1];
      const gradients = currentLayer.computeGradients(
        difference,
        intermediateActivations[i],
        intermediateActivations[i - 1]
      );

      currentLayer.updateWeights(gradients, this.lr);

      // Compute difference for the next layer
      const transposedWeights = currentLayer.weights.transpose();
      difference = difference
        .multiplyWith(transposedWeights)
        .elementWiseMultiplyWith(intermediateActivations[i - 1])
        .dropLastColumn();
    }
  }

  dumpLayers() {
    const layers = this.layers.map((layer) => layer.weights.data);
    return JSON.stringify(layers, null, 2);
  }
}
