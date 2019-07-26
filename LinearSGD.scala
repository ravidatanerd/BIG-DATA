/// Codes referred from https://spark.apache.org/docs/2.2.0/mllib-linear-methods.html

// Libraries Import 
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD

// Load and parse the data
// Also make sure that dataset is loaded in /user/""Username"/ Use : hdfs dfs -put withoutheader.csv /user/"username"/
val data = sc.textFile("withoutheader.csv")
val parsedData = data.map { line => line.split(",").map(_.trim)}

// Separating data set in Label and Features 
val training = parsedData.map { row =>
 val label = (row(9)).toDouble
 val features = Vectors.dense(row(5).toDouble, row(6).toDouble, row(8).toDouble)
 LabeledPoint(label,features)
}.cache()

// Model creation 
val numIterations = 20
val stepSize = 0.001
val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)

// Model Evaluation on dataset and calculating error
val valuesAndPreds = training.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean();
println("Dataset MSE = " + MSE)