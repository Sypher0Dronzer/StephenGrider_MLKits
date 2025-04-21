import loadCSV from "../load-csv.js";
import plot from "node-remote-plot";

import { LogisticRegression } from "./logistic-regression.js";

let { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/cars.csv",
  {
    dataColumns: ["horsepower", "weight", "displacement"],
    labelColumns: ["passedemissions"],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: (value) => {
        return value === "TRUE" ? 1 : 0;
      },
    },
  }
);
const regression = new LogisticRegression(labels, features, {
  iterations: 50,
  learningRate: 0.5,
  batchSize:10,
  decisionBoundary:0.55  
});

regression.train()
console.log(regression.test(testLabels,testFeatures))
// regression.predict([[130,1.75,307],
// [88,1.07,97]]).print()
// console.log(regression.costHistory)

plot({
  x:regression.costHistory.reverse(),
  xLabel: 'Iterations #',
  yLabel:'Cross Entropy or Cost Function'


})