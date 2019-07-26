/// Codes referred from https://spark.apache.org/docs/2.3.0/mllib-linear-methods.html#classification

// Libraries Import 
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._
// Only show log event which has Error
Logger.getLogger("org").setLevel(Level.ERROR)

//Data Frame creation of Logistic Regression

val dataLogR= spark.sql("select Metro_2013 as label,Employed_2016,Unemployed_2016,Median_Household_Income_2016 from unemp_pov ")

// Using Vectore for Putting all independent variables as Features in Vectore form per record
val assembler = new VectorAssembler().setInputCols(Array("Employed_2016","Unemployed_2016","Median_Household_Income_2016")).setOutputCol("features")
val finLogR = assembler.transform(dataLogR).select($"label",$"features")

// Fit the model
val lgr = new LogisticRegression()
val lgrModel = lgr.fit(finLogR)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lgrModel.coefficients} Intercept: ${lgrModel.intercept}")

// Extract the summary from the returned LogisticRegressionModel
val modelSummary = lgrModel.summary

// OutPut 

// objective per iteration
val objectiveHistory = modelSummary.objectiveHistory
println("objective History :")
objectiveHistory.foreach(loss => println(loss))

val binarySummary = modelSummary.asInstanceOf[BinaryLogisticRegressionSummary]

// Receiver-operating characteristic and area under Curve.
val roc = binarySummary.roc
roc.show()
println(s"area under Curve: ${binarySummary.areaUnderROC}")
