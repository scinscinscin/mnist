import { relu, softmax, tanh } from "../src/activations";
import { AveragePoolingLayer, ConvolutionalLayer, FlattenLayer } from "../src/layers/convolutional";
import { FullyConnectedLayer } from "../src/layers/fully_connected";
import { fisher_yates, load_mnist_square } from "./utils/load_mnist";
import { mulberry32 } from "../src/matrix";
import { NeuralNetwork } from "../src/network";
import { ClassificationCrossEntropy1D } from "../src/loss/classification_1d";

let seed = 32;
const samples = load_mnist_square();
const training = samples.filter((e, i) => i <= 4096);
const testing = samples.filter((e, i) => i > 4096 && i < 8192);
const shuffled_training = fisher_yates(training, mulberry32(seed++));

const benchmark = new ClassificationCrossEntropy1D(testing, 10);
const nn = new NeuralNetwork(benchmark);
nn.add_layer(
  new ConvolutionalLayer({ rows: 5, cols: 5, input_channels: 1, output_channels: 6 }, relu, seed++, { padding: 2 }),
)
  .add_layer(new AveragePoolingLayer({ rows: 2, cols: 2 }))
  .add_layer(new ConvolutionalLayer({ rows: 5, cols: 5, input_channels: 6, output_channels: 16 }, relu, seed++))
  .add_layer(new AveragePoolingLayer({ rows: 2, cols: 2 }))
  .add_layer(new FlattenLayer(5, 5, 16))
  .add_layer(new FullyConnectedLayer(400, 120, tanh, seed++))
  .add_layer(new FullyConnectedLayer(120, 84, tanh, seed++))
  .add_layer(new FullyConnectedLayer(84, 10, softmax, seed++));
nn.describe({ rows: 28, columns: 28, channels: 1 });

const learning_rate = 0.05;
const batch_size = 32;
const max_epochs = 100;
nn.train(shuffled_training, { max_epochs, batch_size, learning_rate }).save("weights_cnn.json");
