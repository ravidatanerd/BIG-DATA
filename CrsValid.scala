
// Libraries Import 
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._


// Only show log event which has Error
Logger.getLogger("org").setLevel(Level.ERROR)


// Data Frame creation of Linear Regression
val dataLinR= spark.sql("select POVALL_2016 as label,Employed_2016,Unemployed_2016,Median_Household_Income_2016 from unemp_pov ")

// Using Vectore for Putting all independent variables as Features in Vectore form per record
val assembler = new VectorAssembler().setInputCols(Array("Employed_2016","Unemployed_2016","Median_Household_Income_2016")).setOutputCol("features")

val finLinR = assembler.transform(dataLinR).select($"label",$"features")

// Splitting data in Training and testing set in 90 10 ratio respectively
val Array(training, test) = finLinR.randomSplit(Array(0.9, 0.1), seed = 2018)

val lr = new LinearRegression()

val cvModel = lr.fit(training)


// Output 
println(s"Coefficients: ${cvModel.coefficients} Intercept: ${cvModel.intercept}")
val modelSummary = cvModel.summary
modelSummary.residuals.show()
println(s"RMSE: ${modelSummary.rootMeanSquaredError}")
println(s"MSE: ${modelSummary.meanSquaredError}")
println(s"r2: ${modelSummary.r2}")

// Output for Test Set
cvModel.transform(test).select("features", "label", "prediction").show()



