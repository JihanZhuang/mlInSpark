import scala.reflect.runtime.universe
 
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
 
object TestNaiveBayes {
  
  case class RawDataRecord(category: String, text: String)
  
  def main(args : Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    
    var srcRDD = sc.textFile("/home/jihanzhuang/bigData/testNaiveBayes/sitePageData.data").map { 
      x => 
        var data = x.split(",")
        RawDataRecord(data(0),data(1))
    }
    
    //70%作为训练数据，30%作为测试数据
    val splits = srcRDD.randomSplit(Array(0.5, 0.5))
    var trainingDF = splits(0).toDF()
    var testDF = splits(1).toDF()
    
    //将词语转换成数组
    var tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    var wordsData = tokenizer.transform(trainingDF)
    println("output1：")
    wordsData.select($"category",$"text",$"words").take(1)
    
    //计算每个词在文档中的词频
    var hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
    var featurizedData = hashingTF.transform(wordsData)
    println("output2：")
    featurizedData.select($"category", $"words", $"rawFeatures").take(1)
    
    
    //计算每个词的TF-IDF
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    var idfModel = idf.fit(featurizedData)
    var rescaledData = idfModel.transform(featurizedData)
    println("output3：")
    rescaledData.select($"category", $"features").take(2).foreach(println)
    
	val toDouble = udf[Double, String]( _.toDouble)
	//转换成Bayes的输入格式
    var trainDataRdd = rescaledData.withColumn("label", toDouble(rescaledData("category"))).select($"label",$"features")
    println("output4：")
trainDataRdd.printSchema();
println(trainDataRdd.getClass())

 val tnaiveBayes = new NaiveBayes().setLabelCol("label").setFeaturesCol("features")
    //训练模型
    val model = tnaiveBayes.fit(trainDataRdd)   
    
    //测试数据集，做同样的特征表示及格式转换
    var testwordsData = tokenizer.transform(testDF)
    var testfeaturizedData = hashingTF.transform(testwordsData)
    var testrescaledData = idfModel.transform(testfeaturizedData)
    var testDataRdd = testrescaledData.withColumn("label", toDouble(testrescaledData("category"))).select($"label",$"features")
    
    //对测试数据集使用训练模型进行分类预测
  val predictions = model.transform(testDataRdd)
    predictions.show(3000)  
    //统计分类准确率
    println("output5：")
val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)	
   
  }
}
