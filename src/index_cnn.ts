import { load_mnist_square } from "./load";

async function main() {
  const samples = load_mnist_square();
  const { input, label } = samples[0];

  console.log(input.print());
  console.log(label);
}

main();
