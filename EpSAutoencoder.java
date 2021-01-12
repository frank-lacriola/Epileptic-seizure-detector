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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
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
            .activation(Activation.TANH)
            .l2(0.0001)
            .list()
            .layer(new DenseLayer.Builder().nIn(178).nOut(100)
                .build())
            .layer(new DenseLayer.Builder().nIn(100).nOut(64)
                .build())
            .layer(new DenseLayer.Builder().nIn(64).nOut(100)
                .build())
            .layer(new OutputLayer.Builder().nIn(100).nOut(178)
                .activation(Activation.TANH)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(Collections.singletonList(new ScoreIterationListener(10)));

        //loading data from CSV with schema
        Schema schema = new Schema.Builder()
            .addColumnString("first")
            .addColumnsInteger("int_%d",1,179)
            .build();

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
            .removeColumns("first","int_179").build();

        RecordReader recordReader = new CSVRecordReader(1,',');
        try {
            recordReader.initialize(new FileSplit( new ClassPathResource("data.csv").getFile() ));


        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        //Passing transformation process to convert the csv file
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);
        //Normalizing each batch
        NormalizerMinMaxScaler n = new NormalizerMinMaxScaler(-1,1);
        DataSetIterator iter = new RecordReaderDataSetIterator( transformProcessRecordReader, 100  );
        n.fit(iter);
        iter.setPreProcessor(n);


        List<INDArray> featuresTrain = new ArrayList<>();
        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> labelsTest = new ArrayList<>();
        DataSet dsTest = null;


        Random r = new Random(12345);
        while(iter.hasNext()){
            org.nd4j.linalg.dataset.DataSet ds = iter.next();
            SplitTestAndTrain split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
            featuresTrain.add(split.getTrain().getFeatures());
            dsTest = split.getTest();
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

        }









    }










