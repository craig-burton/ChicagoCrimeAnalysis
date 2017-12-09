package crimes

import scalafx.application.JFXApp
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.FXRenderer
import scala.io.Source



object Chicago extends JFXApp {
  val spark = SparkSession.builder().master("local[*]").getOrCreate()
  import spark.implicits._
  spark.sparkContext.setLogLevel("WARN")

  val file = "crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv"
  val header = Source.fromFile(file).getLines.next().split(',')
  val crimes = spark.read.option("header",true).csv(file)
  println("Total number of crimes" + crimes.count().toString)
  header.foreach(line => {
    crimes.select(line).show(false)
  })

  val notNullCrimes = crimes.filter('Latitude.isNotNull && 'Longitude.isNotNull && 'Latitude > 40)
    .select('Latitude,'Longitude,'FBICode)
  notNullCrimes.show()


  def plotCrimeLatLong(notNullCrimes:DataFrame,title:String,size:Int):Plot = {
    val lat = notNullCrimes.select('Latitude).collect().map(_.getString(0).toDouble)

    val long = notNullCrimes.select('Longitude).collect().map(_.getString(0).toDouble)

    Plot.scatterPlot(lat,long,title,"Latitude","Longitude",size,RedARGB)
  }

  val prostitution = notNullCrimes.filter('FBICode === "16")
  FXRenderer(plotCrimeLatLong(prostitution,"Prostitution in Chicago 2012 to 2017",3))

  val robbery = notNullCrimes.filter('FBICode === "03")
  FXRenderer(plotCrimeLatLong(robbery,"Robbery in Chicago 2012 to 2017",2))

  val homicide = notNullCrimes.filter('FBICode === "01A")
  homicide.show()
  FXRenderer(plotCrimeLatLong(homicide,"Homicide in Chicago 2012 to 2017",4))

  val all3 = notNullCrimes.filter('FBICode === "01A" || 'FBICode === "03" || 'FBICode == "16")
  val cg = ColorGradient(1.0 -> RedARGB, 3.0 -> GreenARGB, 16.0 -> BlackARGB)
  val lat = all3.select('Latitude).collect().map(_.getString(0).toDouble)
  val long = all3.select('Longitude).collect().map(_.getString(0).toDouble)
  val color = all3.select('FBICode).collect().map(_.getString(0)).map(x => {
      if(x == "16") 16.0
      else if(x == "03") 3.0
      else 1.0
    }).map(cg)

  FXRenderer(Plot.scatterPlot(lat,long,"All 3","Lat","Lon",.5,color))


  //Better predictor
  //Clustering
    //Separate predictors in each cluster
  spark.stop()

}
