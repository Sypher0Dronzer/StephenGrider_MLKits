import loadCSV from "../load-csv.js";
import plot from "node-remote-plot";
import _ from 'lodash'
import { LogisticRegression } from "./logistic-regression.js";

let { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/cars.csv",
  {
    dataColumns: ["horsepower","displacement","weight" ],
    labelColumns: ["mpg"],
    shuffle: true,
    splitTest: 50,
    converters: {
      mpg: (value) => {
        const mpg = parseFloat(value)
        if(mpg <15) return [1,0,0]
        else if (mpg>30) return [0,0,1]
        else return [0,1,0]
      },
    },
  }
);
const regression = new LogisticRegression(_.flatMap(labels), features, {
  iterations: 50,
  learningRate: 0.5,
  batchSize:10,
  decisionBoundary:0.55  
});


regression.train()
console.log(regression.test(_.flatMap(testLabels),testFeatures))
regression.predict([[150,200,2.223],[215,440,2.16]]).print()
// console.log(regression.costHistory)

plot({
  x:regression.costHistory.reverse(),
  xLabel: 'Iterations #',
  yLabel:'Cross Entropy or Cost Function'


})