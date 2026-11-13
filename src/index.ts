import { sigmoid, softmax } from "./activations";
import { FullyConnectedLayer } from "./layers/fully_connected";
import { load_mnist } from "./load";
import { NeuralNetwork, Patch, Tensor2D } from "./network";

const samples = load_mnist("./mnist_train_full.csv");
const training = samples.filter((e, i) => i <= 4096);
const testing = samples.filter((e, i) => i > 4096 && i < 8192);

let seed = 32;
const nn = new NeuralNetwork();
nn.add_layer(new FullyConnectedLayer(784, 64, sigmoid, seed++))
  .add_layer(new FullyConnectedLayer(64, 32, sigmoid, seed++))
  .add_layer(new FullyConnectedLayer(32, 10, softmax, seed++));
nn.describe({ rows: 1, columns: 784 });

const LEARNING_RATE = 1.0;
const BATCH_SIZE = 32;

function epoch() {
  let total_loss = 0;
  let count = 0;

  // shuffle indices each epoch (minibatch SGD; the noise is what lets the
  // network escape the "predict the class prior" plateau)
  const idx = training.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }

  for (let b = 0; b < idx.length; b += BATCH_SIZE) {
    const chunk = idx.slice(b, b + BATCH_SIZE);
    let cumulative_patches: Patch[] | null = null;
    let batch_loss = 0;

    for (const i of chunk) {
      const sample = training[i];
      const pred = nn.forward(new Tensor2D(sample.input));

      // cross-entropy: -log(probability assigned to the true class)
      const f = (pred.final as Tensor2D).matrix;
      let loss = 0;
      for (let k = 0; k < f.arr[0].length; k++) if (sample.label.arr[0][k] === 1) loss = -Math.log(f.arr[0][k]);
      batch_loss += loss;

      const sample_update = nn.get_updates(new Tensor2D(sample.label), pred);
      if (cumulative_patches == null) cumulative_patches = sample_update;
      else cumulative_patches = cumulative_patches.map((patch, k) => patch.add(sample_update[k]));
    }

    // get_updates returns updates in reverse order (last layer first), so reverse
    // before zipping onto this.layers (first layer first).
    const normalized = cumulative_patches!.map((e) => e.scale(LEARNING_RATE / chunk.length)).reverse();
    for (let i = 0; i < nn.layers.length; i++) nn.layers[i].update(normalized[i]);

    total_loss += batch_loss;
    count += chunk.length;
  }

  return total_loss / count;
}

let last_overall_pct = null as number | null;
for (let i = 0; i < 100; i++) {
  const t = epoch();
  console.log(`Epoch ${i}. Loss: ${t}`);

  if (i % 10 == 0) {
    const figures = benchmark();
    if (last_overall_pct == null || last_overall_pct <= figures) last_overall_pct = figures;
    else break;
  }
}

nn.save("weights.json");

function benchmark() {
  const num_labels = 10;
  const correct_per_label = new Array(num_labels).fill(0);
  const total_per_label = new Array(num_labels).fill(0);
  let total_correct = 0;

  for (const sample of testing) {
    const forward = nn.forward(new Tensor2D(sample.input));
    const output = (forward.final as Tensor2D).matrix.arr[0];
    const predicted = output.indexOf(Math.max(...output));
    const true_label = sample.label.arr[0].indexOf(1);

    total_per_label[true_label]++;
    if (predicted === true_label) {
      correct_per_label[true_label]++;
      total_correct++;
    }
  }

  console.log("\nBenchmark results:");
  for (let i = 0; i < num_labels; i++) {
    const pct = total_per_label[i] > 0 ? ((correct_per_label[i] / total_per_label[i]) * 100).toFixed(1) : "0.0";
    console.log(`Label ${i}: ${correct_per_label[i]}/${total_per_label[i]} correct (${pct}%)`);
  }
  const overall_pct_n = (total_correct / testing.length) * 100;
  const overall_pct = overall_pct_n.toFixed(1);
  console.log(`Overall accuracy: ${total_correct}/${testing.length} correct (${overall_pct}%)`);
  return overall_pct_n;
}

benchmark();
