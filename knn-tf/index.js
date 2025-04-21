import tf from '@tensorflow/tfjs'
import loadCSV from "./load-csv.js";

function knn(features, labels,predictionPoint,k) {
    // standardization 
    const {mean , variance}= tf.moments(features)
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

  return (
    features
    .sub(mean).div(variance.pow(0.5))
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => a.arraySync()[0] - b.arraySync()[0])
      .slice(0, k)
      .reduce((acc, pairs) => acc + pairs.arraySync()[1], 0) / k
  );
}
let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    dataColumns: ["lat", "long","sqft_lot"],
    labelColumns: ["price"],
    shuffle: true,
    splitTest: 10,
  }
);


features = tf.tensor(features)
labels = tf.tensor(labels)

testLabels.forEach((testPoint,i)=>{

    const result = knn(features,labels,tf.tensor(testPoint), 10)
    const error = (testLabels[i][0] - result )/testLabels[i][0]  * 100 
    console.log("Error: ", error, )
}) 