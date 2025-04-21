import * as tf from "@tensorflow/tfjs";
import loadCSV from "../load-csv.js";
import plot from "node-remote-plot";
import {
  LinearRegression,
  LinearRegressionBasicMethod,
} from "./linear-regression.js";
let { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
  shuffle: true,
  splitTest: 50,
});
const regression = new LinearRegression(labels, features, {
  iterations: 100,
  learningRate: 0.01,
  batchSize:10
});

//  console.log('b4: m =', regression.weights.arraySync()[0],' b=',regression.weights.arraySync()[1])

regression.train();
const r2 = regression.test(testLabels, testFeatures);
regression.predict([[46,.92,97]]).print()
console.log("r2 is", r2);

// plot({
//   x: regression.bHistory.reverse(),
//   y:regression.mseHistory.reverse(),
//   xLabel: "Values of b",
//   yLabel: "Mean Square Error",
// });

plot({
  x:regression.mseHistory.reverse(),
  xLabel: "Iterations",
  yLabel: "Mean Square Error",
});

// console.log(regression.bHistory);

//  console.log('after: m =', regression.weights.arraySync()[1],' b=',regression.weights.arraySync()[0])

//  const oldRegression =new LinearRegressionBasicMethod(features,labels,{iterations:100,learningRate:0.0001})

//  oldRegression.train()
//  console.log('after: m =', oldRegression.m,' b=',oldRegression.b )
