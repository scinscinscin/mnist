import { sigmoid, softmax } from "../src/activations";
import { FullyConnectedLayer } from "../src/layers/fully_connected";
import { fisher_yates, load_mnist } from "./utils/load_mnist";
import { mulberry32 } from "../src/matrix";
import { NeuralNetwork } from "../src/network";
import { ClassificationCrossEntropy1D } from "../src/loss/classification_1d";

let seed = 32;
const samples = load_mnist("./mnist_train_full.csv");
const training = samples.filter((_, i) => i <= 4096);
const testing = samples.filter((_, i) => i > 4096 && i < 8192);
const shuffled_training = fisher_yates(training, mulberry32(seed++));

const learning_rate = 1.0;
const batch_size = 32;
const max_epochs = 100;

const benchmark = new ClassificationCrossEntropy1D(testing, 10);
const nn = new NeuralNetwork(benchmark);

nn.add_layer(new FullyConnectedLayer(784, 64, sigmoid, seed++))
  .add_layer(new FullyConnectedLayer(64, 32, sigmoid, seed++))
  .add_layer(new FullyConnectedLayer(32, 10, softmax, seed++));
nn.describe({ rows: 1, columns: 784 });

nn.train(shuffled_training, { max_epochs, batch_size, learning_rate }).save("weights.json");
