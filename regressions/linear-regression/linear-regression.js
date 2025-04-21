import * as tf from "@tensorflow/tfjs";
import _ from "lodash";
//easy version using arrays
class LinearRegressionBasicMethod {
  constructor(labels, features, options) {
    this.features = features;
    this.labels = labels;

    this.options = Object.assign(
      { learningRate: 0.01, iterations: 1000 },
      options
    );
    this.m = 0;
    this.b = 0;
  }
  gradientDescent() {
    const currentGuessesForMPG = this.features.map((row) => {
      return this.m * row[0] + this.b;
    });

    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return guess - this.labels[i][0];
        })
      ) *
        2) /
      this.features.length;

    const mSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return (guess - this.labels[i][0]) * this.features[i][0];
        })
      ) *
        2) /
      this.features.length;

    this.m = this.m - mSlope * this.options.learningRate;
    this.b = this.b - bSlope * this.options.learningRate;
  }
  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }

  predict() {}
}

class LinearRegression {
  constructor(labels, features, options) {
    this.labels = tf.tensor(labels);

    // this.features = tf.tensor(features);
    // this.features = tf
    //   .ones([this.features.shape[0], 1])
    //   .concat(this.features, 1);

    this.mseHistory = [];
    this.bHistory = [];

    this.features = this.processFeatures(features);
    this.options = Object.assign(
      { learningRate: 0.01, iterations: 1000 },
      options
    );

    // this.m =Math.random();
    // this.b = Math.random();
    this.weights = tf.zeros([this.features.shape[1], 1]);
  }
  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }
  batchGradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }
  train() {
    //BGD
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndexRow = j * this.options.batchSize;
        const featureSlice = this.features.slice(
          [startIndexRow, 0],
          [this.options.batchSize, -1]
        );
        const labelSlice = this.labels.slice(
          [startIndexRow, 0],
          [this.options.batchSize, -1]
        );
        this.batchGradientDescent(featureSlice,labelSlice);
      }
      this.recordMSE();
      this.updateLearningRate();
    }

    // this is for normal gradiest descent
    // for (let i = 0; i < this.options.iterations; i++) {
    //   console.log(this.options.learningRate);
    //   this.bHistory.push(this.weights.arraySync()[0][0]);
    //   this.gradientDescent();
    //   this.recordMSE();
    //   this.updateLearningRate();
    // }
  }
  test(testLabels, testFeatures) {
    testLabels = tf.tensor(testLabels);
    // testFeatures = tf.tensor(testFeatures);
    // testFeatures = tf.ones([testFeatures.shape[0], 1]).concat(testFeatures, 1);
    testFeatures = this.processFeatures(testFeatures);

    const predictions = testFeatures.matMul(this.weights);

    const SSres = testLabels.sub(predictions).pow(2).sum().arraySync();
    const SStotal = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();

    //coeff of determination
    return 1 - SSres / SStotal;
  }

  processFeatures(features) {
    // this will add 1s ti left of the features and test features
    features = tf.tensor(features);

    //basic idea is to use the same mean and variance that is obtained during training , during prediction or testing

    if (this.mean && this.variance) {
      //if its already present means that it has undergone training once and now has those mean and variance so
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      //meaning its in training so can directly call the function
      features = this.standardize(features);
    }
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features);
    this.mean = mean;
    this.variance = variance;
    return features.sub(mean).div(variance.pow(0.5));
  }
  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;
    if (this.mseHistory[0] > this.mseHistory[1]) {
      //means better
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
    // this.mseHistory.pop()
  }

predict(observations){
  return this.processFeatures(observations).matMul(this.weights)
}
}

export { LinearRegression, LinearRegressionBasicMethod };
