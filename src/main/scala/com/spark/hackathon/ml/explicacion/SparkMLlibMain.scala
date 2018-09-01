package com.spark.hackathon.ml.explicacion

import akka.event.slf4j.SLF4JLogging
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.DoubleType

import scala.collection.JavaConversions._


/**
  * Repasar las funcionalidades básicas sobre Pipelines en MLlib de Spark
  */
object SparkMLlibMain extends App with SLF4JLogging {


  /** Creación de la session */

  val configurationProps = ConfigFactory.load().getConfig("spark").entrySet()
    .map(prop => (s"spark.${prop.getKey}", prop.getValue.unwrapped().toString)).toSeq

  val sparkConf = new SparkConf().setAll(configurationProps)
  val sparkSession = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate()

  /** Creación de dataframe */

  println(s"Dataframe canciones ...")

  val canciones = sparkSession.read.json("/home/jcgarcia/hackathon/canciones.json")

  val schema = canciones.schema.add("label", DoubleType)

  val cancionesToFitRDD = canciones.rdd.map { row =>
    if (row.mkString(",").toLowerCase.contains("latin"))
      Row.merge(row, Row(1.0))
    else Row.merge(row, Row(0.0))
  }

  val cancionesToFit = sparkSession.createDataFrame(cancionesToFitRDD, schema)

  cancionesToFit.show()
  cancionesToFit.printSchema()

  // Stages de nuestro workflow o pipeline de ML: tokenizer, hashingTF, and lr.
  val tokenizer = new Tokenizer()
    .setInputCol("genre")
    .setOutputCol("words")
  val hashingTF = new HashingTF()
    .setNumFeatures(1000)
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")
  val lr = new LogisticRegression()
    .setMaxIter(1000)
    .setRegParam(0.001)
  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, lr))

  // Entrenamos nuestro pipeline con los datos de entrenamiento para generar un modelo
  val model = pipeline.fit(cancionesToFit)

  // Guardamos el modelo en disco (HDFS) para luego productivizarlo
  model.write.overwrite().save("/tmp/spark-logistic-regression-model")

  // El pipeline tambien podemos guardarlo para recuperarlo en otro momento, no esta entrenado!
  pipeline.write.overwrite().save("/tmp/unfit-lr-model")

  // Podemos cargar el modelo para productivizarlo ....
  val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")

  // Documentos para predecir si son latinos o no
  val test = sparkSession.createDataFrame(Seq(
    ("corazon espinado", "Mana", "Unplugged", "Pop"),
    ("no ha parado de llover", "Mana", "Unplugged", "Pop"),
    ("sin pijama", "Becky-G", "Vete tu a saber", "Latin"),
    ("felices los cuatro", "Maluma", "Tiene discos?", "Latin power")
  )).toDF("song", "artist", "album", "genre")

  // Predicciones con el modelo guardado
  model.transform(test)
    .select("song", "artist", "album", "genre", "probability", "prediction")
    .collect()
    .foreach { case Row(song: String, artist: String, album: String, genre: String, prob: Vector, prediction: Double) =>
      log.error(s"($song, $artist, $album, $genre) --> prob=$prob, prediction=$prediction")
    }

}

