import { relu, sigmoid, softmax } from "../src/activations";
import { mulberry32 } from "../src/matrix";
import { LSTMLayer } from "../src/layers/lstm";
import { NeuralNetwork, Tensor3D } from "../src/network";
import { fisher_yates } from "./utils/load_mnist";
import { load_pos } from "./utils/load_pos";
import { SequentialFullyConnectedLayer } from "../src/layers/sequential";
import { ClassificationCrossEntropy2D } from "../src/loss/classification_2d";

const W2V_EMBEDDING_SIZE = 200;
const HIDDEN_SIZE = 128;

const { samples, tags } = load_pos();

let seed = 33;
const training = samples.filter((e, i) => i <= 4096);
const testing = samples.filter((e, i) => i > 4096 && i < 8192);
const shuffled_training = fisher_yates(training, mulberry32(seed++));

const benchmark = new ClassificationCrossEntropy2D(testing, tags.length, 5);
const lstm = new NeuralNetwork(benchmark)
  .add_layer(new LSTMLayer(W2V_EMBEDDING_SIZE, HIDDEN_SIZE, seed++))
  .add_layer(new SequentialFullyConnectedLayer(HIDDEN_SIZE, tags.length, softmax, seed++));

const EXAMPLE_WORD_COUNT = 15;
lstm.describe({ rows: 1, columns: W2V_EMBEDDING_SIZE, channels: EXAMPLE_WORD_COUNT });

const learning_rate = 0.5;
const batch_size = 32;
const max_epochs = 49;

lstm.train(shuffled_training, { max_epochs, batch_size, learning_rate }).save("weights_pos_lstm.json");

console.log("\nFinal network accuracy");
benchmark.benchmark(lstm);
