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
 /*   val conf = new SparkConf()
val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
 
//将原始数据映射到DataFrame中，字段category为分类编号，字段text为分好的词，以空格分隔
var srcDF = sc.textFile("/home/jihanzhuang/bigData/testNaiveBayes/test/test.txt").map { 
      x => 
        var data = x.split(",")
        RawDataRecord(data(0),data(1))
}.toDF()
srcDF.select("category", "text").take(2).foreach(println)
//将分好的词转换为数组
var tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
var wordsData = tokenizer.transform(srcDF)
wordsData.select($"category",$"text",$"words").take(2).foreach(println)
var hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100)
var featurizedData = hashingTF.transform(wordsData)	
featurizedData.select($"category", $"words", $"rawFeatures").take(2).foreach(println)
var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
var idfModel = idf.fit(featurizedData)
var rescaledData = idfModel.transform(featurizedData)
rescaledData.select($"features").take(2).toArray.foreach(println)
//转换成Bayes的输入格式
    var trainDataRdd = rescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }
    println("output4：")
println(trainDataRdd.getClass())*/
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
/*import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext,SparkConf}

object naiveBayes {
  def main(args: Array[String]) {
    val conf =new SparkConf()
    val sc =new SparkContext(conf)

    //读入数据
    val data = sc.textFile(args(0))
    val parsedData =data.map { line =>
      val parts =line.split(",",2)
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(",").map(_.toDouble)))
    }
    // 把数据的60%作为训练集，40%作为测试集.
    //val splits = parsedData.randomSplit(Array(0.9,0.1),seed = 11L)
    //val training =splits(0)
    //val test =splits(1)

    //获得训练模型,第一个参数为数据，第二个参数为平滑参数，默认为1，可改
    val model =NaiveBayes.train(parsedData,lambda = 1.0)

    //对模型进行准确度分析
    val predictionAndLabel= parsedData.map(p => (model.predict(p.features),p.label,model.predictProbabilities(p.features)))
    val accuracy =1.0 *predictionAndLabel.filter(x => x._1 == x._2).count() / parsedData.count()
	//predictionAndLabel.collect().foreach(println)
	val tmp1=predictionAndLabel.collect();
	val tmp2=parsedData.collect();
	for(i<-0 until tmp1.length){println(i+"-------->"+tmp2(i)+"-------->"+tmp1(i))}
    println("accuracy-->"+accuracy)
    //println("Predictionof (0.0, 2.0, 0.0, 1.0):"+model.predict(Vectors.dense(6,0.202985233541,0.205069820129,53.0151348493,0,12)))
    //println("Predictionof (0.0, 2.0, 0.0, 1.0):"+model.predictProbabilities(Vectors.dense(6,0.202985233541,0.205069820129,53.0151348493,0,12)))
  }
}

*/
