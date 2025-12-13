const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const sigmoidPrime = (x: number) => x * (1 - x);

const relu = (x: number) => (x > 0 ? x : 0);
const reluPrime = (x: number) => (x > 0 ? 1 : 0.00001);

const softplus = (x: number) => Math.log(1 + Math.exp(x));
const softplusPrime = (x: number) => 1 / (1 + Math.exp(-x));

const linear = (x: number) => x;
const linearPrime = (x: number) => 1;

export const Activations = {
  sigmoid: { f: sigmoid, prime: sigmoidPrime },
  relu: { f: relu, prime: reluPrime },
  softplus: { f: softplus, prime: softplusPrime },
  linear: { f: linear, prime: linearPrime },
};

export type ActivationFunction = (typeof Activations)[keyof typeof Activations];
