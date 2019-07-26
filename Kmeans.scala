/// Codes referred from Professor: Austin Henslee, Kmeans_example.scala

// Libraries Import 
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.IntegerType
import spark.implicits._
import org.apache.log4j._

// Only show log event which has Error
Logger.getLogger("org").setLevel(Level.ERROR)

{
// Data Frame creation of Linear Regression
val dataLinR= spark.sql("select POVALL_2016,Employed_2016,Unemployed_2016,Median_Household_Income_2016 from unemp_pov ") 
val k = 9
for (k <- 2 to 15)
{
  val assembler = new VectorAssembler().setInputCols(Array("POVALL_2016","Employed_2016","Unemployed_2016","Median_Household_Income_2016")).setOutputCol("featureVector")
  val data = assembler.transform(dataLinR)
  println("Output from assembler.transform:")
  data.show()

  val kmeans = new KMeans()
        .setPredictionCol("cluster")
        .setFeaturesCol("featureVector")
        .setK(k)
        .setInitSteps(40)
        .setMaxIter(99)

  val kmModel = kmeans.fit(data)

  println("Cluster centroids:")
  kmModel.clusterCenters.foreach(println)

  println(s"$k,${kmModel.computeCost(data)}")

  val predictions = kmModel.summary.predictions
  predictions.orderBy("cluster","POVALL_2016","Employed_2016","Unemployed_2016","Median_Household_Income_2016").show()
  predictions.count()
}
}
