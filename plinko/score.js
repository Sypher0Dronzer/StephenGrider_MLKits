let outputs = [];
const k=10
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 10;

  // let numberCorrect = 0;
  // for (let i = 0; i < testSet.length; i++) {
  //   const bucket = knn(trainingSet, testSet[i][0]);
  //   if (bucket == testSet[i][3]) {
  //     numberCorrect++;
  //   }
  // }
  // console.log("Accuracy :" + (numberCorrect / testSetSize).toFixed(2));

// in this we are taking range of values of K
  // _.range(1, 20).forEach((k) => {
  //   const accuracy = _.chain(testSet)
  //     .filter(testpoint => knn(trainingSet, _.initial(testpoint), k) == testpoint[3])
  //     .size()
  //     .divide(testSetSize)
  //     .value();

  //   console.log("Accuracy for ",k," :" , accuracy);
  // });

  _.range(0, 3).forEach((feature) => {
    // position or bounciness or size individually to select feature
    const data = outputs.map(row => [row[feature] , _.last(row)])
  const [testSet, trainingSet] = splitDataset(minMax(data,1), testSetSize);

    const accuracy = _.chain(testSet)
      .filter(testpoint => knn(trainingSet, _.initial(testpoint), k) == _.last(testpoint))
      .size()
      .divide(testSetSize)
      .value();

    console.log("Accuracy for feature ",feature," :" , accuracy);
  });
}
function knn(dataset, point, k) {
  // we need the point to have 3 elements only so that it works when we manually want to test the prediction as well
  //last gives the last element of the array
  // initial gives all elements except the last 
  return _.chain(dataset)
    .map((row) =>{ 
      return [distance(_.initial(row), point),
         _.last(row)]
    }
  )
    .sortBy((row) => row[0])
    .slice(0, k)
    .countBy((row) => row[1])
    .toPairs()
    .sortBy((row) => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}
function distance(pointA, pointB) {

  //pointA =[200,0.55,16]
  //pointB =[300,0.5,16]
  
  _.chain(pointA)
    .zip(pointB)
    .map(([a, b]) => Math.pow(a - b, 2))
    .sum()
    .value() ** 0.5;
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}

//this function helps to normalise
function minMax(data,featureCount){
  const clonedData =_.cloneDeep(data)

  for(let i=0;i<featureCount;i++){
    const column = clonedData.map(row => row[i])

    const min = _.min(column)
    const max = _.max(column)

    for(let j=0;j<clonedData.length;j++){
      clonedData[j][i]= (clonedData[j][i] - min) / (max-min)
    }
  }
  return clonedData
}