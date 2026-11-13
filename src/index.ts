import { sigmoid, softmax } from "./activations";
import { FullyConnectedLayer } from "./layers/fully_connected";
import { fisher_yates, load_mnist } from "./load";
import { mulberry32 } from "./matrix";
import { ClassificationCrossEntropyBenchmark, NeuralNetwork, Patch, Tensor2D } from "./network";

let seed = 32;
const nn = new NeuralNetwork();
nn.add_layer(new FullyConnectedLayer(784, 64, sigmoid, seed++))
  .add_layer(new FullyConnectedLayer(64, 32, sigmoid, seed++))
  .add_layer(new FullyConnectedLayer(32, 10, softmax, seed++));
nn.describe({ rows: 1, columns: 784 });

const samples = load_mnist("./mnist_train_full.csv");
const training = samples.filter((_, i) => i <= 4096);
const testing = samples.filter((_, i) => i > 4096 && i < 8192);
const shuffled_training = fisher_yates(training, mulberry32(seed++));

const learning_rate = 1.0;
const batch_size = 32;
const max_epochs = 100;
const benchmark = new ClassificationCrossEntropyBenchmark(testing, 10);
nn.train(shuffled_training, { max_epochs, batch_size, learning_rate, benchmark }).save("weights.json");
