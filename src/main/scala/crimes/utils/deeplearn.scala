package utils

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.lossfunctions.LossFunctions
import scala.io._
import java.io._


object DeepLearnCrime extends App {

  val locationDescriptionsMap = Array("RAILROAD PROPERTY", "AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA", "POLICE FACILITY/VEH PARKING LOT", "MOTEL", "SIDEWALK", "PUBLIC GRAMMAR SCHOOL", "AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA", "CTA GARAGE / OTHER PROPERTY", "CAR WASH", "TRUCKING TERMINAL", "AIRPORT/AIRCRAFT", "HOSPITAL", "MEDICAL/DENTAL OFFICE", "FEDERAL BUILDING", "TRAILER", "SCHOOL, PUBLIC, GROUNDS", "CTA STATION", "SPORTS ARENA/STADIUM", "HOUSE", "CEMETARY", "ROOMING HOUSE",   "VACANT LOT", "SCHOOL, PRIVATE, BUILDING",
    "DRIVEWAY", "VEHICLE-COMMERCIAL", "COUNTY JAIL", "APPLIANCE STORE", "WAREHOUSE", "AIRPORT TERMINAL UPPER LEVEL - SECURE AREA", "AIRPORT EXTERIOR - NON-SECURE AREA", "VEHICLE - OTHER RIDE SERVICE", "DUMPSTER", "COIN OPERATED MACHINE", "CTA PLATFORM", "BARBER SHOP/BEAUTY SALON", "CLUB", "null", "GANGWAY", "BANK", "FACTORY/MANUFACTURING BUILDING", "CHA GROUNDS", "GROCERY FOOD STORE", "BRIDGE", "RESIDENCE-GARAGE", "CHA STAIRWELL", "CONVENIENCE STORE", "LAKEFRONT/WATERFRONT/RIVERBANK", "PUBLIC HIGH SCHOOL",
    "ATM (AUTOMATIC TELLER MACHINE)", "BASEMENT", "AUTO", "ELEVATOR", "JUNK YARD/GARBAGE DUMP", "AIRCRAFT", "HOTEL", "CHA HALLWAY/STAIRWELL/ELEVATOR", "TAXI CAB", "STREET", "DRIVEWAY - RESIDENTIAL", "LAKE", "CHURCH", "RIVER", "BARBERSHOP", "SCHOOL YARD", "ATHLETIC CLUB", "GARAGE/AUTO REPAIR", "OTHER COMMERCIAL TRANSPORTATION", "TRUCK", "CHA ELEVATOR", "COMMERCIAL / BUSINESS OFFICE", "LIBRARY", "LIVERY AUTO", "RIVER BANK",
    "CTA BUS", "CTA BUS STOP", "LAUNDRY ROOM", "GAS STATION DRIVE/PROP.", "SCHOOL, PUBLIC, BUILDING", "CREDIT UNION", "BANQUET HALL", "GOVERNMENT BUILDING", "NURSING HOME", "GAS STATION", "CHA HALLWAY", "CTA L TRAIN", "CTA L PLATFORM", "VEHICLE NON-COMMERCIAL", "SMALL RETAIL STORE", "CHA APARTMENT", "ABANDONED BUILDING", "TAVERN/LIQUOR STORE", "CLEANERS/LAUNDROMAT", "YARD", "CHA LOBBY", "RESIDENCE", "CHA PLAY LOT", "BOAT/WATERCRAFT", "PAWN SHOP",
    "POOLROOM", "POOL ROOM", "OFFICE", "AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA", "STAIRWELL", "SCHOOL, PRIVATE, GROUNDS", "AIRPORT VENDING ESTABLISHMENT", "DRUG STORE", "DEPARTMENT STORE", "VACANT LOT/LAND", "HOSPITAL BUILDING/GROUNDS", "FACTORY", "CURRENCY EXCHANGE", "CHURCH PROPERTY", "CHA BREEZEWAY", "ALLEY", "AIRPORT BUILDING NON-TERMINAL - SECURE AREA", "PARKING LOT", "PARK PROPERTY", "RESIDENCE PORCH/HALLWAY", "CLEANING STORE", "AIRPORT TRANSPORTATION SYSTEM (ATS)", "RESTAURANT", "FOREST PRESERVE", "RESIDENTIAL YARD (FRONT/BACK)",
    "NEWSSTAND", "APARTMENT", "OTHER", "DAY CARE CENTER", "RETAIL STORE", "TAVERN", "CTA TRAIN", "HALLWAY", "AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA", "HOTEL/MOTEL", "PORCH", "ANIMAL HOSPITAL", "GOVERNMENT BUILDING/PROPERTY", "FUNERAL PARLOR", "LIQUOR STORE", "TAXICAB", "MOVIE HOUSE/THEATER", "OTHER RAILROAD PROP / TRAIN DEPOT", "CHA PARKING LOT/GROUNDS", "LOADING DOCK", "COACH HOUSE", "AIRPORT TERMINAL LOWER LEVEL - SECURE AREA", "COLLEGE/UNIVERSITY GROUNDS", "YMCA", "FIRE STATION",
    "CHA PARKING LOT", "SAVINGS AND LOAN", "AIRPORT EXTERIOR - SECURE AREA", "JAIL / LOCK-UP FACILITY", "VEHICLE - DELIVERY TRUCK", "BOWLING ALLEY", "CONSTRUCTION SITE", "SEWER", "CTA PROPERTY", "DELIVERY TRUCK", "PRAIRIE", "HIGHWAY/EXPRESSWAY", "BAR OR TAVERN", "COLLEGE/UNIVERSITY RESIDENCE HALL", "GARAGE", "WOODED AREA", "PARKING LOT/GARAGE(NON.RESID.)", "NURSING HOME/RETIREMENT HOME", "CTA TRACKS - RIGHT OF WAY", "VESTIBULE", "CHURCH/SYNAGOGUE/PLACE OF WORSHIP", "AIRPORT PARKING LOT", "LIVERY STAND OFFICE", "EXPRESSWAY EMBANKMENT", "LAGOON")
    .map(x => x.replace("\"","")).zipWithIndex.toMap

  val fbiCodes = Map(
    "Homicide" -> "01A",
    "Criminal Sexual Assault" -> "02",
    "Robbery" -> "03",
    "Aggravated Assault" -> "04A",
    "Aggravated Battery" -> "04B",
    "Burglary" -> "05",
    "Larceny" -> "06",
    "Motor Vehicle Theft" -> "07",
    "Arson" -> "09",
    "Involuntary Manslaughter" -> "01B",
    "Simple Assault" -> "08A",
    "Simple Battery" -> "08B",
    "Forgery & Counterfeiting" -> "10",
    "Fraud" -> "11",
    "Embezzlement" -> "12",
    "Stolen Property" -> "13",
    "Vandalism" -> "14",
    "Weapons Violation" -> "15",
    "Prostitution" -> "16",
    "Criminal Sexual Abuse" -> "17",
    "Drug Abuse" -> "18",
    "Gambling" -> "19",
    "Offenses Against Family" -> "20",
    "Liquor License" -> "22",
    "Disorderly Conduct" -> "24",
    "Misc Non-Index Offense" -> "26"
  )

  val file1 = "crimes-in-chicago/Chicago_Crimes_2001_to_2004.csv"
  val file2 = "crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv"
  val file3 = "crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv"
  val file4 = "crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv"
  val fbiMap = fbiCodes.map(_._2).zipWithIndex.toMap
  val headerMap = Source.fromFile(file1).getLines.next().split(',').zipWithIndex.toMap
  val colsToKeep = "LocationDescription Beat District Ward CommunityArea FBICode XCoordinate YCoordinate Year".split(" ")
  val colsMap = colsToKeep.zipWithIndex.toMap


  def handleLine(line:String):String = {
    val arrayData = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)")
    if(arrayData.contains("")) {
      ""
    } else {
      val str:String = colsToKeep.map(x => {
        if(x == "LocationDescription") {
          val key = arrayData(headerMap(x)).replace("\"","")
          if(key == "" || key == "Description") key else locationDescriptionsMap(key).toString
        } else if(x == "FBICode") {
          fbiMap(arrayData(headerMap(x))).toString
        } else {
          arrayData(headerMap(x))
        }
      }).foldLeft("")((agg,next) => agg + "," + next)
      str.slice(1,str.length())
    }
  }

  //If empty string exists, delete it
  def concatFiles(names:Array[String],filename:String) {
    if(!(new File(filename).exists())) {
      println("Creating files")
      val fw = new FileWriter(filename,true)
      var count = 0
      names.foreach(str => {
        val lines = Source.fromFile(str).getLines
        //Skip the header
        lines.next()
        while(lines.hasNext) {
          val line = lines.next()
          val newLine = handleLine(line)
          if(newLine != "") {
            count = count + 1
            fw.write(newLine + "\n")

          }
        }
        println("Finished file : " + str)
        println("Line count : " + count)
      })
      fw.close()
      println("Final line count : " + count)
    } else {
      println("already existed " + filename)
    }
  }

  val filenames = Array(file1,file2,file3,file4)
  val newFile = "crimes-in-chicago/Chicago_Crime_Data.csv"
  concatFiles(filenames,newFile)


  //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
  var numLinesToSkip = 0
  var delimiter = ','
  var recordReader = new CSVRecordReader(numLinesToSkip,delimiter)
  recordReader.initialize(new FileSplit(new File(newFile)))

  //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
  var labelIndex = 5;     //9 values in each row of the iris.txt CSV: 8 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
  var numClasses = 26;     //26 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
  var batchSize = 100000;    //Using a batch size

  var iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
  var allData = iterator.next();
  allData.shuffle();
  var testAndTrain = allData.splitTestAndTrain(0.8);  //Use 80% of data for training

  var trainingData = testAndTrain.getTrain();
  var testData = testAndTrain.getTest();

  //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
  var normalizer = new NormalizerStandardize();
  normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
  normalizer.transform(trainingData);     //Apply normalization to the training data
  normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


  var numInputs = 8;
  var outputNum = 26;
  var iterations = 1000;
  var midNum = 100;
  var midNum2 = 100;
  var midNum3 = 20;
  var midNum4 = 1;
  var seed = 6;


  println("Build model....");
  var conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation(Activation.TANH)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .learningRate(0.1)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(midNum)
          .build())
      .layer(1, new DenseLayer.Builder().nIn(midNum).nOut(midNum2)
          .build())
      .layer(2, new DenseLayer.Builder().nIn(midNum2).nOut(midNum3)
          .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.L2)
          .activation(Activation.SIGMOID)
          .nIn(midNum3).nOut(outputNum).build())
      .backprop(true).pretrain(false)
      .build();

  //run the model
  var model = new MultiLayerNetwork(conf);
  model.init();
  model.setListeners(new ScoreIterationListener(iterations));

  model.fit(trainingData);

  //evaluate the model on the test set
  var eval = new Evaluation(outputNum);
  var output = model.output(testData.getFeatureMatrix());
  eval.eval(testData.getLabels(), output);
  println(eval.stats())

}
