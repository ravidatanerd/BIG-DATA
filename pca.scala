/// Codes referred from https://spark.apache.org/docs/2.3.0/mllib-dimensionality-reduction.html

// Libraries Import 
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._


// Only show log event which has Error
Logger.getLogger("org").setLevel(Level.ERROR)

// Data Frame creation of Linear Regression
val df= spark.sql("select Civilian_labor_force_2016,Employed_2016,Unemployed_2016,Unemployment_rate_2016,Median_Household_Income_2016 from unemp_pov ")

// Using Vectore for Putting all independent variables as Features in Vectore form per record
val assembler = new VectorAssembler().setInputCols(Array("Civilian_labor_force_2016","Employed_2016","Unemployed_2016","Unemployment_rate_2016","Median_Household_Income_2016")).setOutputCol("features")

val finalDf = assembler.transform(df).select($"features")

println("Output from assembler.transform:")
finalDf.show()

val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3).fit(finalDf)
val pcaDF = pca.transform(finalDf)
val result = pcaDF.select("pcaFeatures")
println("Output of top 3 principal components:")
result.show(false)
