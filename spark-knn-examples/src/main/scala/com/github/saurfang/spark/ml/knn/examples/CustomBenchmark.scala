package com.github.saurfang.spark.ml.knn.examples

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{KNNClassifier, NaiveKNNClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.tuning.{Benchmarker, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, SQLContext}
import org.rogach.scallop._

import scala.collection.mutable

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val k = opt[Int](default = Some(5), descr = "Number of neighbours")
  val sample = opt[Int](default = None, descr = "Run on sample")
  val dataset = opt[String](default = Some("mnist"), descr = "mnist or svhn")
  val partitions = opt[Int](default = Some(10), descr = "Number of partitions")
  verify()
}

/**
  * Benchmark KNN as a function of number of observations
  */
object CustomBenchmark {

  def main(args: Array[String]) {
    val conf = new Conf(args)
    val trainingPath = conf.dataset() match {
      case "mnist" => "data/mnist"
      case "svhn" => "data/SVHN"
    }

    val testingPath = conf.dataset() match {
      case "mnist" => "data/mnist.t"
      case "svhn" => "data/SVHN.t"
    }

    val numPartitions = conf.partitions()

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    val rawTrainingDataset = MLUtils.loadLibSVMFile(sc, trainingPath)
      .zipWithIndex()
      .sortBy(_._2, numPartitions = numPartitions)
      .keys
      .toDF()

    // convert "features" from mllib.linalg.Vector to ml.linalg.Vector
    val training = MLUtils.convertVectorColumnsToML(rawTrainingDataset)
      .cache()
    training.count() //force persist

    val rawTestingDataset = MLUtils.loadLibSVMFile(sc, testingPath)
      .zipWithIndex()
      .sortBy(_._2, numPartitions = numPartitions)
      .keys
      .toDF()

    // convert "features" from mllib.linalg.Vector to ml.linalg.Vector
    val testing = MLUtils.convertVectorColumnsToML(rawTestingDataset)
      .cache()
    testing.count() //force persist

    val knn = new KNNClassifier()
      .setTopTreeSize(numPartitions * 10)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setK(1)

    val knnModel = knn.fit(training)
    val predicted = knnModel.transform(training)
  }
}
