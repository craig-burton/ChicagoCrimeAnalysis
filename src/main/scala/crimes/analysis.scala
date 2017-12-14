package crimes

import scalafx.application.JFXApp
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.FXRenderer
import scala.io.Source

import utils.ClassifyCrimes

object Chicago extends JFXApp {
  val spark = SparkSession.builder().master("local[*]").getOrCreate()
  import spark.implicits._
  spark.sparkContext.setLogLevel("WARN")

  val dateUDF = udf((s:String) => {
    //Array of month, day, year
    Array(s.substring(0,2).toInt,s.substring(3,5).toInt,s.substring(6,10).toInt)
  })

  val districtConcatUDF = udf((fst:String,scd:String,thd:String) => {
    fst + " " + scd + " " + thd
  })

  val fbiCodes = Map(
    "Homicide" -> "01A",
    "Criminal Sexual Assault" -> "02",
    "Robbery" -> "03",
    "Aggravated Assault" -> "04A",
    "Aggravated Battery" -> "04B",
    "Burglary" -> "05",
    "Larceny" -> "06",
    "Motor Vehicle Theft" -> "07",
    "Arson" -> "09",
    "Involuntary Manslaughter" -> "01B",
    "Simple Assault" -> "08A",
    "Simple Battery" -> "08B",
    "Forgery & Counterfeiting" -> "10",
    "Fraud" -> "11",
    "Embezzlement" -> "12",
    "Stolen Property" -> "13",
    "Vandalism" -> "14",
    "Weapons Violation" -> "15",
    "Prostitution" -> "16",
    "Criminal Sexual Abuse" -> "17",
    "Drug Abuse" -> "18",
    "Gambling" -> "19",
    "Offenses Against Family" -> "20",
    "Liquor License" -> "22",
    "Disorderly Conduct" -> "24",
    "Misc Non-Index Offense" -> "26"

  )

  val fbiFlipped = fbiCodes.map(_.swap)

  val file1 = "crimes-in-chicago/Chicago_Crimes_2001_to_2004.csv"
  val file2 = "crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv"
  val file3 = "crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv"
  val file4 = "crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv"

  val header = Source.fromFile(file1).getLines.next().split(',')
  val crimes1 = spark.read.option("header",true).csv(file1)
  val crimes2 = spark.read.option("header",true).csv(file2)
  val crimes3 = spark.read.option("header",true).csv(file3)
  val crimes4 = spark.read.option("header",true).csv(file4)
  val crimesNoYear = crimes1.union(crimes2.union(crimes3.union(crimes4)))
  val crimes = crimesNoYear.withColumn("udfArray",dateUDF(col("Date")))
                    .withColumn("Month",col("udfArray")(0))
                    .withColumn("Day",col("udfArray")(1))
                    .withColumn("Year",col("udfArray")(2))
                    .withColumn("DistrictMonth",districtConcatUDF(col("District"),col("Month"),col("Year")))
                    .drop("udfArray")


  val notNullCrimes = crimes.filter('Latitude.isNotNull && 'Longitude.isNotNull && 'Latitude > 40)



  def plotCrimeLatLong(notNullCrimes:DataFrame,title:String,size:Int):Plot = {
    val lat = notNullCrimes.select('Latitude).collect().map(_.getString(0).toDouble)

    val long = notNullCrimes.select('Longitude).collect().map(_.getString(0).toDouble)

    Plot.scatterPlot(lat,long,title,"Latitude","Longitude",size,RedARGB)
  }

  // val prostitution = notNullCrimes.filter('FBICode === "16")
  // FXRenderer(plotCrimeLatLong(prostitution,"Prostitution in Chicago 2012 to 2017",3))
  //
  // val robbery = notNullCrimes.filter('FBICode === "03")
  // FXRenderer(plotCrimeLatLong(robbery,"Robbery in Chicago 2012 to 2017",2))
  //
  // val homicide = notNullCrimes.filter('FBICode === "01A")
  // homicide.show()
  // FXRenderer(plotCrimeLatLong(homicide,"Homicide in Chicago 2012 to 2017",4))
  //
  // val all3 = notNullCrimes.filter('FBICode === "01A" || 'FBICode === "03" || 'FBICode == "16")
  // val cg = ColorGradient(1.0 -> RedARGB, 3.0 -> GreenARGB, 16.0 -> BlackARGB)
  // val lat = all3.select('Latitude).collect().map(_.getString(0).toDouble)
  // val long = all3.select('Longitude).collect().map(_.getString(0).toDouble)
  // val color = all3.select('FBICode).collect().map(_.getString(0)).map(x => {
  //     if(x == "16") 16.0
  //     else if(x == "03") 3.0
  //     else 1.0
  //   }).map(cg)
  //
  // FXRenderer(Plot.scatterPlot(lat,long,"All 3","Lat","Lon",.5,color))


  def plotCountYearByYear(crimes:DataFrame,fbiCode:String,crimeType:String):Plot = {
    plotAllByYear(crimes.filter('FBICode === fbiCode),"Count of " + crimeType + " by Year")
  }

  def plotAllByYear(crimes:DataFrame,title:String):Plot = {
    val data:Array[(Double,Double)] = crimes.groupBy('Year)
                    .agg(count('FBICode).as("Count"))
                    .collect().map(row => row.getInt(0).toDouble->row.getLong(1).toDouble/1000)
                    .sortBy(_._1)
    Plot.scatterPlotWithLines(data.map(_._1),data.map(_._2),title,"Year","Count (thousands)",.2,lineGrouping=1)

  }
  //
  // fbiCodes.foreach(tup => {
  //   FXRenderer(plotCountYearByYear(notNullCrimes,tup._2,tup._1))
  // })

  // FXRenderer(plotAllByYear(notNullCrimes,"All Crime Count by Year"))
  // notNullCrimes.groupBy('FBICode).agg(count('Year).as("Count"))
  //   .orderBy(-'Count)
  //   .take(6).foreach(println)


  //Classifying FBI Code
  val stringFeatureCols = "LocationDescription CommunityArea Arrest Ward FBICode".split(" ")
  //TODO: Add the time of day, in a 24 hour thing
  val intFeatureCols:Array[String] = "Year Day Month".split(" ")
  val notNullFeatures = notNullCrimes.select('Block, 'LocationDescription, 'Arrest, 'CommunityArea,
    'Beat, 'Ward, 'District, 'Year, 'FBICode, 'Day, 'Month).filter(row => !row.anyNull)
  // val assembledData = ClassifyCrimes.assembleData(notNullFeatures,intFeatureCols,stringFeatureCols,"FBICode",true)
  // println(ClassifyCrimes.classifyDecisionTreeMultiClass(assembledData,false))
  // println(ClassifyCrimes.classifyRandomForestMultiClass(assembledData,20,5,true))
  //



  //Classifying Arrest made
  // val assembledArrest = ClassifyCrimes.assembleData(notNullFeatures,intFeatureCols,stringFeatureCols,"Arrest",true)
  // println(ClassifyCrimes.classifyDecisionTreeBinary(assembledArrest,true))
  // println(ClassifyCrimes.classifyRandomForestBinary(assembledArrest,20,5,true))
  //TODO: importance of features w/ relative weight in RandomForestClassifier
  //RandomForestClassificationModel.featureImportances()
  //Look at relative rates of false positives/false negatives
  //When are you wrong, are you predicting more arrests or less arrests




  //How many arrests?
  // val arrested = notNullCrimes.filter('Arrest === "True")
  // val notArrested = notNullCrimes.filter('Arrest === "False")
  // println("Arrested: " + arrested.count())
  // println("Not arrested: " + notArrested.count())

  // val arrestedGroups = arrested.groupBy('FBICode).agg(count('Year).as("Count")).orderBy(-'Count)
  // arrestedGroups.show(false)
  // val notArrestedGroups = notArrested.groupBy('FBICode).agg(count('Year).as("Count")).orderBy(-'Count)
  // notArrestedGroups.show(false)
  //
  // val arrestedToPlot = arrestedGroups.collect().map(row => fbiFlipped(row.getString(0)) -> row.getLong(1).toDouble)
  // val barArrested = Plot.barPlot(arrestedToPlot.map(_._1.slice(0,10)).toSeq,Seq(arrestedToPlot.map(_._2).toSeq->RedARGB),false,0.8,"Number of Arrests Made by Crime Type","Crime Type","Number of Arrests")
  // FXRenderer(barArrested)
  //
  // val notArrestedToPlot = notArrestedGroups.collect().map(row => fbiFlipped(row.getString(0)) -> row.getLong(1).toDouble)
  // val barNotArrested = Plot.barPlot(notArrestedToPlot.map(_._1.slice(0,10)).toSeq,Seq(arrestedToPlot.map(_._2).toSeq->RedARGB),false,0.8,"Number of No Arrests Made by Crime Type","Crime Type","Number of No Arrests")
  // FXRenderer(barNotArrested)

  //Recommend how many crimes will occur on a given day in a given area(District?)
  // notNullCrimes.groupBy('DistrictMonth).agg(count('Year).as("Count")).orderBy(-'Count).show()
  //
  // println(notNullCrimes.groupBy('Block).agg(count('Year)).count())

  //Cluster the location of the crimes?
  // val locAssemble = new VectorAssembler().setInputCols(Array("LatDouble","LonDouble","Month")).setOutputCol("Loc")
  // val doubleLatLon = notNullCrimes.withColumn("LatDouble",'Latitude.cast(DoubleType))
  //                                 .withColumn("LonDouble",'Longitude.cast(DoubleType))
  //                                 .filter(row => !row.anyNull)
  //                                 .filter('FBICode === "16")
  // val locationsCrimes = locAssemble.transform(doubleLatLon)
  // val kMeans = new KMeans().setK(3).setFeaturesCol("Loc")
  // val clusterModel = kMeans.fit(locationsCrimes)
  // val clusters = clusterModel.transform(locationsCrimes)
  // clusters.show()
  //
  // val x = clusters.select('LonDouble).as[Double].collect()
  // val y = clusters.select('LatDouble).as[Double].collect()
  // val predict = clusters.select('prediction).as[Double].collect()
  // val cg = ColorGradient(0.0 -> RedARGB, 1.0 -> GreenARGB, 2.0 -> BlueARGB)
  // val plot = Plot.scatterPlot(x, y, "Crime Clusters", "Longitude", "Latitude", 3, predict.map(cg))
  // FXRenderer(plot)
  //TODO: Cluster prostitution data

  println(notNullCrimes.select('FBICode).distinct.count())


  //Better predictor
  //Clustering
    //Separate predictors in each cluster
  spark.stop()

}
