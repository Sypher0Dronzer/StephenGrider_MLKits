import * as tf from "@tensorflow/tfjs";

class LogisticRegression {
  constructor(labels, features, options) {
    this.labels = tf.tensor(labels);

    this.costHistory = []; // cross entropy 
    this.bHistory = [];

    this.features = this.processFeatures(features);
    this.options = Object.assign(
      { learningRate: 0.01, iterations: 1000,decisionBoundary:0.55 },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }
  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights).sigmoid();
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
      this.recordCost();
      this.updateLearningRate();
    }

    // this is for normal gradiest descent
    // for (let i = 0; i < this.options.iterations; i++) {
    //   this.bHistory.push(this.weights.arraySync()[0][0]);
    //   this.gradientDescent();
    //   this.recordCost();
    //   this.updateLearningRate();
    // }
  }
  test(testLabels, testFeatures) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    const incorrect = predictions.sub(testLabels).abs().sum().arraySync();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
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
  recordCost() {
    const guesses= this.features.matMul(this.weights).sigmoid()

    const termOne= this.labels.transpose().matMul(guesses.log())

    const termTwo = this.labels
    .mul(-1)
    .add(1)
    .transpose()
    .matMul(
        guesses.mul(-1).add(1).log()
    ) 

   const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).arraySync()[0][0]
   this.costHistory.unshift(cost)
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) return;
    if (this.costHistory[0] > this.costHistory[1]) {
      //means better
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
    // this.mseHistory.pop()
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary);
  }
}

export { LogisticRegression };
