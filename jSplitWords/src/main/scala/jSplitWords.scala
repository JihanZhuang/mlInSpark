import scala.reflect.runtime.universe

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object jSplitWords {
  def main(args : Array[String]) {
	val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    var srcRDD = sc.textFile("/home/jihanzhuang/bigData/testNaiveBayes/sitePageData.data").map {
      x =>
        var data = x.split(",")
        RawDataRecord(data(0),data(1),data(2),data(3))
    }

    //70%作为训练数据，30%作为测试数据
    val splits = srcRDD.randomSplit(Array(0.8, 0.2))
    var trainingDF = splits(0).toDF()
    trainingDF.select($"text").flatMap(case Row(text: String)=>{
      val wordMaxLen=5;
      for(i<-0 to wordMaxLen){
        for(j<-0 to text.length()){

        }
      }
    })
  }
}
