import { Network } from "../src/network";
import { Matrix } from "../src/matrix";
import { RNG } from "../src/rng";
import { loadData } from "./dataloader";
import { directEncoder } from "../src/utils/labelEncoders";

const network = new Network({
  lr: 0.001,
  rng: new RNG(100),
  layers: [
    { size: 3, activation: "linear" },
    { size: 3, activation: "linear" },
    { size: 1, activation: "linear" },
  ],
});

async function main() {
  console.log("Loading samples...");
  const encoder = directEncoder();
  const samples = await loadData("test/linear.csv", encoder);
  console.log("Loaded samples!");

  let i = 0;
  while (true) {
    const sample = samples[i % samples.length];
    const prediction = network.forward(sample.inputs);

    const difference = sample.label.subtractWith(prediction.result);
    const loss = Math.abs(difference.average());
    console.log(`loss at ${i}: ${loss}`);

    if (loss < 0.00000000000001) break;
    network.backward(sample.label, [sample.inputs, ...prediction.intermediateActivations]);
    i++;
  }

  console.log("Trained network:");
  const { result: finalOutput } = network.forward(Matrix.fromData([[100, 100, 100]]));
  console.log(finalOutput);

  const dump = network.dumpLayers();
  console.log(dump);
}

main();
