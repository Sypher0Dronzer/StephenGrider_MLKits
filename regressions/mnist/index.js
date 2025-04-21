import plot from "node-remote-plot";
import _ from "lodash";
import { LogisticRegression } from "./logistic-regression.js";
import mnist from "mnist-data";

const mnistData = mnist.training(0, 60000);

const features = mnistData.images.values.map((e) => _.flatMap(e));

const encodedLabels = mnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 80,
  batchSize: 50,
});

regression.train();

const testMnistData= mnist.testing(0,1000)
const testFeatures= testMnistData.images.values.map((e) => _.flatMap(e));
const testEncodedLabels =testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});
const accuracy = regression.test(testEncodedLabels,testFeatures)
console.log('Accuracy is ',accuracy.toFixed(3))
plot({
  x:regression.costHistory.reverse(),
  xLabel: 'Iterations #',
  yLabel:'Cross Entropy or Cost Function'


})