/**
  * Created by jihanzhuang on 2016/11/1.
  */
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.sql.Row

import scala.collection.mutable.{ArrayBuffer, ListBuffer, Map}
import scala.reflect.io.{File, Path}
object jSplitWords {
  case class RawDataRecord(category: String,siteId:String,page:String, text: String)

  def main(args : Array[String]) {
    /*var test="我和你新联";
   var tmp=splitString(test,5)
    tmp.foreach(println)
    return*/
    val conf = new SparkConf().setMaster("local").setAppName("jSplitWords").set("spark.sql.crossJoin.enabled","true")
    val sc = new SparkContext(conf)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    var srcRdd = sc.textFile("C:\\Users\\jihanzhuang\\Desktop\\sitePageData1.data").map {
      x =>
        var data = x.split(",")
        RawDataRecord(data(0),data(1),data(2),data(3))
    }
    /*//存放(String,Int)结构。
    var words=srcRdd.map(row=>{
      var arr=collection.mutable.ArrayBuffer[(String,Int)]()
      for(i<-0 to row.text.length-1){
        for(j<-1 to 5){
            arr+=((row.text.substring(i,Math.min(i+j,row.text.length-1)),1))
        }
      }
      arr
    })*/
    /*var words=srcRdd.flatMap(row=>{
      var arr=Map[String,Tuple2[Map[String,Int],Map[String,Int]]]()
      var wordCount=5;
      for(i<-0 to row.text.length-1-wordCount){
        for(j<-1 to wordCount){
            arr++=Map(row.text.substring(i,Math.min(i+j,row.text.length-1))->Tuple2(Map("st"->1),Map("sdf"->1)))
        }
      }
      arr
    })
    words.collect().foreach(println)
    var wordsCount=words.groupBy(row=>row._1).mapValues(_.size)*/
    /*var rdd=srcRdd.map(row=>
      {
        var tmp=row.text.replaceAll(","," ")
        tmp
      }
    )
    rdd.getClass
    rdd.foreach(println)*/
    //var test="送阿桑菲尼"
    //左信息和右信息
    var words=srcRdd.flatMap(row=>{
      var arr=ArrayBuffer[(String,String,String,Int)]()
      var wordCount=5
      for(i<-0 to row.text.length-1){
        for(j<-1 to wordCount){
          if(i==0){
            if(i+j<row.text.length) {
              arr += ((row.text.substring(i, i + j) , " ",row.text.substring(i+j,i+j+1),1))
            }
            if(i+j==row.text.length){
              arr += ((row.text.substring(i, i + j) , " "," ",1))
            }
          }else {
            if(i+j==row.text.length) {
              arr += ((row.text.substring(i, i + j) ,row.text.substring(i - 1, i)," ",1))
            }
            if(i+j<row.text.length){
              arr += ((row.text.substring(i, i + j) ,row.text.substring(i - 1, i),row.text.substring(i+j,i+j+1),1))
            }
          }
        }
      }
      arr
    })
    words.sortBy(row=>row._1.length).foreach(println)
    var wordsDF=words.toDF("word","left","right","count")
    var leftDoc=wordsDF.select("word","left").rdd//.sortBy(row=>(row.apply(1).toString,true,1)).foreach(println)
    println("左信息聚合")
    var wordsLeft=leftDoc.groupBy(row=>row.apply(0).toString).map(row=>{
      var data=row._2;
      var leftMap=ListBuffer[(String ,Int)]()
      for(item <- data){
        leftMap+=((item.apply(1).toString,1))
      }
      leftMap.getClass
      var tmp=leftMap.groupBy(row=>row._1).mapValues(_.map(_._2).sum)
      //var tmp=rdd.map(word=>(word,1)).reduceByKey(_+_)
      (row._1,tmp)
    })
    //wordsLeft.foreach(println)
    //Info Entropy
    var wordsLeftIE=wordsLeft.map(row=>{
      var leftMap=row._2
      var sum=leftMap.map(_._2).sum.toDouble
      var tmp=leftMap.map(_._2).map(value=>{
        var info=0.0
          info=(value.toDouble/sum)*Math.log(value/sum)
        info
      })
      var leftInfo=tmp.sum*(-1)
      (row._1,leftInfo)
    })
    wordsLeftIE.sortBy(row=>row._1.length).foreach(println)
    var wordsCount=wordsDF.select("word","count").rdd
    //wordsCount.getClass
    var eachWordCount=wordsCount.map({case Row(word:String,count:Int)=>word->count}).reduceByKey(_+_)
    var sum=eachWordCount.map(_._2).sum
    println(sum)
    var eachWordCountDF=eachWordCount.toDF("word","count")
    var eachWordCountDF1=eachWordCount.toDF("word","count")
    var wordJoin=eachWordCountDF.join(eachWordCountDF1,eachWordCountDF.col("word").contains(eachWordCountDF1.col("word"))).toDF("word","count","contain","ccount")
    var wordJoinRdd=wordJoin.rdd.groupBy(row=>row.apply(0))
    wordJoinRdd.foreach(println)
    wordJoinRdd.map(row=>row._2).foreach(println)
    /*wordJoinRdd.map{row=>{
      var word=row._1
      var part=row._2
      //var tmp=1.0/sum
      //tmp
      var wordCombination=splitString(word.toString,word.toString.length)
      for(i<-0 to wordCombination.size){

      }
    }}*///.foreach(println)
    /*var sumCount=words.count()
    var wordsCount=words.reduceByKey(_+_)
    wordsCount.collect().foreach(println)
    var wordsCountDf1=wordsCount.toDF("name","count")
    var wordsCountDf2=wordsCount.toDF("name1","count1")
    var wordsJoin=wordsCountDf1.join(wordsCountDf2,wordsCountDf1.col("name").contains(wordsCountDf2.col("name1"))).map{
      case Row(name:String,count:Int,name1:String,count1:Int)=>(name,(count,name1,count1))
    }
    wordsJoin.collect().foreach(println)*/
  }
  def splitString(str:String,left:Int): ArrayBuffer[String] ={
      var strList=ArrayBuffer[String]()
      if(left==0){
        return strList
      }
    if(str.length==0){
        return strList
    }else{
        var start=str.length-left;
        for(i<-1 to left){
          if(start+i<=str.length) {
            var tmpStr = ","+str.substring(start, start + i)
            var tmpList=splitString(str.substring(start+i),left-start-i)
            if(tmpList.size==0){
              strList+=tmpStr
            }else {
              for (j <- 0 to tmpList.size-1) {
                strList += tmpStr + tmpList.apply(j)
              }
            }
            //strList+=tmpStr//splitString(str.substring(start+i),left-1)
          }
        }
        strList
      }
  }
}