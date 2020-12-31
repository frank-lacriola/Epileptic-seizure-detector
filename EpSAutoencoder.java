package EpSAutoencoder;

import org.bytedeco.javacpp.tools.Logger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.*;

public class EpSAutoencoder {

    public static void main(String[] args){


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(new AdaGrad(0.05))
            .activation(Activation.RELU)
            .l2(0.0001)
            .list()
            .layer(new DenseLayer.Builder().nIn(178).nOut(100)
                .build())
            .layer(new DenseLayer.Builder().nIn(100).nOut(64)
                .build())
            .layer(new DenseLayer.Builder().nIn(64).nOut(100)
                .build())
            .layer(new OutputLayer.Builder().nIn(100).nOut(178)
                .activation(Activation.LEAKYRELU)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(Collections.singletonList(new ScoreIterationListener(10)));

        //loading data from CSV with schema
        Schema schema = new Schema.Builder()
            .addColumnString("first")
            .addColumnsInteger("qqq","iqb", "mcw", "mut", "pzd",
                "cvq", "day", "otn", "thd", "dax", "cgk", "fmm" ,"vwt", "cev", "eji",
                "mtr","fvj" ,"wqh" ,"hpl" ,"ppl", "kgo", "eux" , "jwm" ,"qae", "jnj","gwn", "rby", "cat" ,"pet" ,"ayo", "okq","wen" ,"qkr" ,"frb", "bmy",
                "jmk", "hnk" ,"ngu" ,"fml", "wpd", "uky", "lju" ,"qmm", "llq" ,"cdc" ,"ato", "abi", "jag", "sdf", "zep", "hck", "mtv", "auk", "ann" ,"clt",
                "mrm" ,"iug" ,"roq" ,"vtg", "rfi" ,"esz", "mci" ,"ftg", "ipg" ,"ert" ,"ohm", "kov", "cpm", "jsm", "ppc", "vpm", "llx", "acu", "zhx", "usi",
                "lce" ,"qql" ,"zwi" ,"ugt", "njs" ,"fit", "bfy" ,"hio", "mhj" ,"drk" ,"ooj", "bry", "wwk", "jjz" ,"eud", "rwc" ,"bkl", "zth", "dbm" ,"qcx" ,"enc" ,"bjx" ,"snx",
                "tkj" ,"lxp" ,"iac" ,"eno", "bsg" ,"ivq", "rqt" ,"kkj", "ybf" ,"jck" ,"gja", "cwe", "kff", "bxr" ,"lze" ,"hkb" ,"jkr", "gra", "tkg", "uox" ,"bgm" ,"plx" ,"qte",
                "hlf" ,"fxp" ,"pmo" ,"gik", "jjj" ,"xey", "qmc" ,"yka", "fpn" ,"inc" ,"hqx", "fic", "arp", "tga", "fnc", "jap" ,"yzq" ,"fcs", "cle", "eik" ,"fum" ,"yvq" ,"uqc",
                "hbs" ,"hoe" ,"wky" ,"jvc", "rda" ,"feh", "hts" ,"xzb", "qcu" ,"mpt" ,"uqq", "hgn" ,"fpg" ,"ueo", "skk" ,"xzj" ,"ito" ,"jqx", "bpd", "gpr" ,"rgi" ,"btw" ,"jys",
                "iib" ,"vhk", "cks" ,"gok", "czl" ,"xtd", "pnn" ,"jnf", "ooe" ,"rfx" ,"mjf", "last")
            .build();

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
            .removeColumns("first","last").build();

        RecordReader recordReader = new CSVRecordReader(1,',');
        try {
            recordReader.initialize(new FileSplit( new ClassPathResource("data.csv").getFile() ));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess); //Passing transformation process to convert the csv file


        DataSetIterator iter = new RecordReaderDataSetIterator( transformProcessRecordReader, 100  );

        List<INDArray> featuresTrain = new ArrayList<>();
        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> labelsTest = new ArrayList<>();

        Random r = new Random(12345);
        while(iter.hasNext()){
            org.nd4j.linalg.dataset.DataSet ds = iter.next();
            SplitTestAndTrain split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
            featuresTrain.add(split.getTrain().getFeatures());
            org.nd4j.linalg.dataset.DataSet dsTest = split.getTest();
            featuresTest.add(dsTest.getFeatures());
            INDArray indexes = Nd4j.argMax(dsTest.getLabels(),1); //Convert from one-hot representation -> index
            labelsTest.add(indexes);
        }

        //Training the model:
        int nEpochs = 30;
        for( int epoch=0; epoch<nEpochs; epoch++ ){
            for(INDArray data : featuresTrain){
                net.fit( data,data);
            }
            System.out.println("Epoch " + epoch + " complete");
        }

        //Evaluation of the model

        }









    }









}
