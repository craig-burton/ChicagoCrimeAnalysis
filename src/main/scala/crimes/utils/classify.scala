package utils

import org.apache.spark.sql.Encoders
import org.apache.spark.ml.classification.{RandomForestClassifier,DecisionTreeClassifier,DecisionTreeClassificationModel,RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator,MulticlassClassificationEvaluator}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import scala.io.Source



object ClassifyCrimes {

  def assembleData(df:DataFrame,intFeatureCols:Array[String],stringFeatureCols:Array[String],labelCol:String,labelColIsString:Boolean):DataFrame = {
    val labelFinal = if(labelColIsString) labelCol+"-i" else labelCol
    val withLabel = stringFeatureCols.foldLeft(df) {(ds, name) =>
      val indexer = new StringIndexer().setInputCol(name)
                                      .setOutputCol(name+"-i")
      indexer.fit(ds).transform(ds)
    }.withColumnRenamed(labelFinal,"label")
    (intFeatureCols ++ stringFeatureCols.map(_+"-i").filter(x=>x!=labelFinal)).zipWithIndex.foreach(println)
    val va = new VectorAssembler().setInputCols(intFeatureCols ++ stringFeatureCols.map(_+"-i").filter(x=>x!=labelFinal))
                    .setOutputCol("features")
    va.transform(withLabel)
  }

  def classifyDecisionTreeMultiClass(assembledData:DataFrame,print:Boolean):Double = {
    //Get data together
    val Array(trainData,testData) = assembledData.randomSplit(Array(.8,.2))
    val dt = new DecisionTreeClassifier().setMaxBins(34188)
    val model = dt.fit(trainData)
    val predictions = model.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator
    val accuracy = evaluator.evaluate(predictions)
    if(print) {
      println("Tree Model:")
      val treeModel = model.asInstanceOf[DecisionTreeClassificationModel]
      println(treeModel.toDebugString)
    }
    accuracy
  }

  def classifyDecisionTreeBinary(assembledData:DataFrame,print:Boolean):Double = {
    //Get data together
    val Array(trainData,testData) = assembledData.randomSplit(Array(.8,.2))
    val dt = new DecisionTreeClassifier().setMaxBins(34188)
    val model = dt.fit(trainData)
    val predictions = model.transform(testData)
    val evaluator = new BinaryClassificationEvaluator
    val accuracy = evaluator.evaluate(predictions)
    if(print) {
      println("Tree Model:")
      val treeModel = model.asInstanceOf[DecisionTreeClassificationModel]
      println(treeModel.toDebugString)
    }
    accuracy
  }

  def classifyRandomForestMultiClass(assembledData:DataFrame,numTrees:Int,maxDepth:Int,print:Boolean):Double = {
    val Array(trainData,testData) = assembledData.randomSplit(Array(.8,.2))
    val rf = new RandomForestClassifier().setMaxBins(34188)
                          .setNumTrees(numTrees)
                          .setMaxDepth(maxDepth)
    val model = rf.fit(trainData)
    val predictions = model.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator
    if(print) {
      println("Tree Model:")
      val treeModel = model.asInstanceOf[DecisionTreeClassificationModel]
      println(treeModel.toDebugString)
      val featureImportances = model.featureImportances.toArray.zipWithIndex
      featureImportances.foreach(println)
    }
    evaluator.evaluate(predictions)
  }

  def classifyRandomForestBinary(assembledData:DataFrame,numTrees:Int,maxDepth:Int,print:Boolean):Double = {
    val Array(trainData,testData) = assembledData.randomSplit(Array(.8,.2))
    val rf = new RandomForestClassifier().setMaxBins(34188)
                        .setNumTrees(numTrees)
                        .setMaxDepth(maxDepth)
    val model = rf.fit(trainData)
    val predictions = model.transform(testData)
    val falsePositives = predictions.filter(col("prediction") === 1.0 && col("label") === 0.0)
    println("False Positives: " + falsePositives.count())
    val falseNegatives = predictions.filter(col("prediction") === 0.0 && col("label") === 1.0)
    println("False Negatives: " + falseNegatives.count())
    val evaluator = new BinaryClassificationEvaluator
    if(print) {
      println("Tree Model Features:")
      val featureImportances = model.featureImportances.toArray.zipWithIndex
      featureImportances.foreach(println)
    }
    evaluator.evaluate(predictions)
  }

}
