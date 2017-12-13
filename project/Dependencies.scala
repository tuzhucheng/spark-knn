import sbt._
import Keys._

object Dependencies {
  val Versions = Seq(
    crossScalaVersions := Seq("2.11.8", "2.10.6"),
    scalaVersion := crossScalaVersions.value.head
  )

  object Compile {
    val breeze_natives = "org.scalanlp" %% "breeze-natives" % "0.12" % "provided"

    object Test {
      val scalatest = "org.scalatest" %% "scalatest" % "2.2.4" % "test"
      val sparktest = "org.apache.spark" %% "spark-core" % "2.2.0"  % "test" classifier "tests"
      val scallop = "org.rogach" %% "scallop" % "3.1.1"
    }
  }

  import Compile._
  import Test._
  val l = libraryDependencies

  val core = l ++= Seq(scalatest, sparktest, scallop)
  val examples = core +: (l ++= Seq(breeze_natives))
}
