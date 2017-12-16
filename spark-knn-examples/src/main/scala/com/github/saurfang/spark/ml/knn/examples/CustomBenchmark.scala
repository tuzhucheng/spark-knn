package com.github.saurfang.spark.ml.knn.examples

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.KNNClassificationModel
import org.apache.spark.ml.classification.{KNNClassifier, NaiveKNNClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.tuning.{Benchmarker, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
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

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1000000.0 + "ms")
    result
  }

  def main(args: Array[String]) {
    val conf = new Conf(args)
    val k = conf.k()

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    import spark.implicits._

    val trainingPath = conf.dataset() match {
      case "mnist" => "data/mnist.bz2"
      case "svhn" => "data/SVHN.bz2"
    }

    val testingPath = conf.dataset() match {
      case "mnist" => "data/mnist.t.bz2"
      case "svhn" => "data/SVHN.t.bz2"
    }

    val numPartitions = conf.partitions()

    val rawTrainingDF = MLUtils.loadLibSVMFile(sc, trainingPath)
    val trainingNumFeats = rawTrainingDF.take(1)(0).features.size
    val rawTrainingDataset = rawTrainingDF
      .zipWithIndex()
      .sortBy(_._2, numPartitions = numPartitions)
      .keys
      .toDF()


    // convert "features" from mllib.linalg.Vector to ml.linalg.Vector
    val training = MLUtils.convertVectorColumnsToML(rawTrainingDataset)
      .cache()
    training.count() //force persist

    val rawTestingDataset = MLUtils.loadLibSVMFile(sc, testingPath).map(p => {
      val sparseVec = p.features.toSparse
      val features = new SparseVector(trainingNumFeats, sparseVec.indices, sparseVec.values)
      new LabeledPoint(p.label, features)
    }).zipWithIndex()
      .sortBy(_._2, numPartitions = numPartitions)
      .keys
      .toDF()

    // convert "features" from mllib.linalg.Vector to ml.linalg.Vector
    val testing = MLUtils.convertVectorColumnsToML(rawTestingDataset)
      .cache()
    testing.count() //force persist

    var knnModel: Option[KNNClassificationModel] = None
    time {
      val knn = new KNNClassifier()
        .setTopTreeSize(numPartitions * 10)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setK(k)

      knnModel = Some(knn.fit(training))
      println("Training finished")
    }

    time {
      val predicted = knnModel.get.transform(testing)

      predicted.createOrReplaceTempView("predicted")
      val correctCount = spark.sql("SELECT count(*) FROM predicted WHERE label == prediction").collect()(0).getLong(0)
      val accuracy = correctCount * 1.0 / predicted.count
      println($"Test set accuracy: ${accuracy}")
    }

  }
}
