import { ActivationFunction } from "./activations";
import { Matrix } from "./matrix";

type RandomFloatFn = () => number;

interface LayerOptions {
  input: number;
  output: number;
  activationFunction: ActivationFunction;
  rng?: RandomFloatFn;
}

export class Layer {
  weights: Matrix;
  activationFunction: ActivationFunction;
  input: number;
  output: number;

  constructor(options: LayerOptions) {
    this.input = options.input;
    this.output = options.output;
    this.activationFunction = options.activationFunction;

    this.weights = Matrix.fromRandom(this.input + 1, this.output, () => (options.rng ?? Math.random)());
  }

  public static fromWeights(options: { weights: Matrix; activationFunction: ActivationFunction }) {
    const layer = new Layer({
      input: options.weights.rows - 1,
      output: options.weights.columns,
      activationFunction: options.activationFunction,
    });

    layer.weights = options.weights;
    return layer;
  }

  forward(input: Matrix) {
    const rawActivations = Matrix.fromZeros(input.rows, this.output);

    for (let i = 0; i < input.rows; i++) {
      for (let j = 0; j < this.output; j++) {
        let sum = 0;

        for (let k = 0; k < input.columns + 1; k++) {
          // +1 because of the bias term
          const x = k === input.columns ? 1 : input.data[i][k];
          sum += x * this.weights.data[k][j];
        }

        rawActivations.data[i][j] = sum;
      }
    }

    const result = rawActivations.mapWith(this.activationFunction.f);
    return { rawActivations, result };
  }

  computeGradients(computedError: Matrix, rawActivations: Matrix, inputs: Matrix) {
    const gradients = Matrix.fromZeros(this.weights.rows, this.weights.columns);

    for (let i = 0; i < this.weights.rows; i++) {
      for (let j = 0; j < this.weights.columns; j++) {
        let gradientSum = 0;

        for (let k = 0; k < computedError.rows; k++) {
          const inputActivation = i === this.weights.rows - 1 ? 1 : inputs.data[k][i];
          const sensitivity = this.activationFunction.prime(rawActivations.data[0][j]);
          gradientSum += computedError.data[k][j] * sensitivity * inputActivation;
        }

        gradients.data[i][j] = gradientSum;
      }
    }

    return gradients;
  }

  updateWeights(gradients: Matrix, learningRate: number) {
    this.weights = this.weights.addWith(gradients.mapWith((x) => x * learningRate));
  }
}
