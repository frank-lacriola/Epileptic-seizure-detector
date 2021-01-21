package EpSAutoencoder;

import com.google.flatbuffers.FlatBufferBuilder;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.exception.Nd4jNoSuchWorkspaceException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.shade.protobuf.compiler.PluginProtos;

import java.io.File;
import java.io.IOException;
import java.nio.LongBuffer;
import java.util.*;

public class EpSAutoencoder {

    public static void main(String[] args) throws IOException {


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
        net.setListeners(Collections.singletonList(new ScoreIterationListener(100)));

        //loading data from CSV with schema
        Schema schema = new Schema.Builder()
            .addColumnString("first")
            .addColumnsInteger("int_%d", 1, 179)
            .build();

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
            .integerMathOp("int_179", MathOp.Subtract, 1)
            .removeColumns("first").build();

        RecordReader recordReader = new CSVRecordReader(1, ',');
        try {
            recordReader.initialize(new FileSplit(new ClassPathResource("data.csv").getFile()));


        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        //Passing transformation process to convert the csv file
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader, transformProcess);

        int batchSize= 100;

        //Normalizing each batch
        NormalizerMinMaxScaler n = new NormalizerMinMaxScaler(-1, 1);
        DataSetIterator iter = new RecordReaderDataSetIterator(transformProcessRecordReader, batchSize, 178, 5);
        n.fit(iter);
        iter.setPreProcessor(n);


        List<INDArray> featuresTrain = new ArrayList<>();
        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> normLabelsTest = new ArrayList<>();
        DataSet dsTest = null;


        Random r = new Random(12345);
        while (iter.hasNext()) {

            org.nd4j.linalg.dataset.DataSet ds = iter.next();
            SplitTestAndTrain split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
            featuresTrain.add(split.getTrain().getFeatures());
            dsTest = split.getTest();
            featuresTest.add(dsTest.getFeatures());
            normLabelsTest.add(ds.getLabels());
        }

/*
        //Creazione vettore per labels
        List<NDArray> labels = new ArrayList<>();
        NDArray batchLabArray = new NDArray();
        batchLabArray.setShapeAndStride();
        for (int i = 0; i < normLabelsTest.size(); i++) {
            for (int j = 0; j < batchSize; j++) {
                batchLabArray.add(Nd4j.argMax((normLabelsTest.get(i)).getRow(j)));  //Reverse one-hot-repr. transformation to get the real label
            }
            labels.add(batchLabArray);
        }
*/
        //Training of the model:
        int nEpochs1 = 15;
        for (int epoch = 0; epoch < nEpochs1; epoch++) {
            for (INDArray data : featuresTrain) {
                net.fit(data, data);

            }
            System.out.println("Epoch " + epoch + " complete");
        }


        System.out.println(net.summary());


        //Initializing RecordWriter to create a CSV with new input files
        List<Writable> writingList = new ArrayList<>();
        File tempFile = File.createTempFile("writeOn", ".csv");
        FileSplit fileSplit = new FileSplit(tempFile);
        CSVRecordWriter recordWriter = new CSVRecordWriter();

        try {
            recordWriter.initialize(fileSplit , new NumberOfRecordsPartitioner());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }


        //Layer reduction and new DataSet after features reduction
        MultiLayerNetwork redNet = new TransferLearning.Builder(net)
            .removeLayersFromOutput(2)
            .setFeatureExtractor(1)
            .build();
        System.out.println(redNet.summary());


     //   List<INDArray> features = new ArrayList<>();
     //   List<INDArray> newLabels = new ArrayList<>();
        List<INDArray> newInputs = new ArrayList<>();



/*
        while (iter2.hasNext()) {
            org.nd4j.linalg.dataset.DataSet ds = iter2.next();
            features.add(ds.getFeatures());
            newLabels.add(ds.getLabels());
        }

*/

        for (INDArray redData : featuresTest) {
            newInputs = redNet.feedForward(redData);
        }


        //Writing of CSV file
          for (int i=0; i<newInputs.size(); i++){
              for (int j=0; j<batchSize ;j++){
              writingList.add(new Text(newInputs.get(i).getRow(j) +","+normLabelsTest.get(i).getRow(j)));
              }
          }

        recordWriter.write(writingList);

/*

        //Inizializing classifier
        MultiLayerConfiguration classifierConfig = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(new AdaGrad(0.05))
            .activation(Activation.RELU)
            .l2(0.0001)
            .list()
            .layer(new DenseLayer.Builder().nIn(64).nOut(32)
                .build())
            .layer(new DenseLayer.Builder().nIn(32).nOut(16)
                .build())
            .layer(new DenseLayer.Builder().nIn(16).nOut(8)
                .build())
            .layer(new OutputLayer.Builder().nIn(8).nOut(2)
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .build())
            .build();

        MultiLayerNetwork classifier = new MultiLayerNetwork(classifierConfig);



        //int nEpochs2 = 30;

      /*  for (int j=0; j< nEpochs2; j++){
            while(inputsIter.hasNext() && labelsIter.hasNext()){
            classifier.fit( (INDArray) inputsIter.next(), (INDArray) labelsIter.next());
            classifier.score();
            }
        }

*/


    }
}











