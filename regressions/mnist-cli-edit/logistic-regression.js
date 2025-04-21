import * as tf from "@tensorflow/tfjs";

class LogisticRegression {
  constructor(features, labels, testLabels, testFeatures, options) {
    this.labels = tf.tensor(labels);

    this.testLabels = testLabels;
    this.testFeatures = testFeatures;

    this.costHistory = []; // cross entropy
    this.bHistory = [];
    
    this.features = this.processFeatures(features);
    this.options = Object.assign(
      { learningRate: 0.01, iterations: 1 },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }
  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }
  batchGradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.options.learningRate));
  }
  train() {
    //BGD
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndexRow = j * this.options.batchSize;

        //tidy tensorflow
        this.weights=tf.tidy(()=>{

          const featureSlice = this.features.slice(
            [startIndexRow, 0],
            [this.options.batchSize, -1]
          );
          const labelSlice = this.labels.slice(
            [startIndexRow, 0],
            [this.options.batchSize, -1]
          );
         return  this.batchGradientDescent(featureSlice, labelSlice);
        })
      }
      this.test(this.testLabels, this.testFeatures);
      console.log("Iteration ", i, " Accuracy is ", this.accuracy); 
      this.recordCost();
      this.updateLearningRate();
    }
  }
  test(testLabels, testFeatures) {
    const predictions = this.predictTest(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);
    // predictions.print();
    // testLabels.print();
    const incorrect = predictions.notEqual(testLabels).sum().arraySync();

    this.accuracy = (predictions.shape[0] - incorrect) / predictions.shape[0];
    return this.accuracy;
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
    const filler = variance.cast("bool").logicalNot().cast("float32");
    this.mean = mean;
    this.variance = variance.add(filler);
    return features.sub(mean).div(this.variance.pow(0.5));
  }
  recordCost() {
    const cost =tf.tidy(()=>{

      const guesses = this.features.matMul(this.weights).softmax();
  
      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
  
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1)
        .add(1e-7) // adding a constant to prevernt log(0)
        .log());
  
      return  termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .arraySync()[0][0];
    })

    this.costHistory.unshift(cost);
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
    const processed = this.processFeatures(observations)
      .matMul(this.weights)
      .softmax();

    const classIndices = processed.argMax(1);
    return tf.oneHot(classIndices, this.labels.shape[1]);
  }
  predictTest(observations) {
    const processed = this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);

    return processed;
  }
}

export { LogisticRegression };
