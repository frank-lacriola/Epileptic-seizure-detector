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
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.util.Index;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.INDArrayIterator;
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
import org.nd4j.linalg.indexing.IntervalIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.shade.protobuf.compiler.PluginProtos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.LongBuffer;
import java.util.*;

public class EpSAutoencoder {

    private static Logger log = LoggerFactory.getLogger("Log");


    public static void main(String[] args) throws Exception {

        Random r = new Random(12345);

        int batchSize = 100;
        boolean trained = true;
        if (!trained) {

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


            //Normalizing each batch
            NormalizerMinMaxScaler n = new NormalizerMinMaxScaler(-1, 1);

            DataSetIterator iter = new RecordReaderDataSetIterator.Builder(transformProcessRecordReader, batchSize)
                .classification(178, 5)
                .build();
            n.fit(iter);
            iter.setPreProcessor(n);


            List<INDArray> featuresTrain = new ArrayList<>();
            List<INDArray> featuresTest = new ArrayList<>();
            List<INDArray> labelsTrain = new ArrayList<>();
            List<INDArray> labelsTest = new ArrayList<>();


            while (iter.hasNext()) {

                DataSet ds = iter.next();
                SplitTestAndTrain split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
                featuresTrain.add(split.getTrain().getFeatures());
                featuresTest.add(split.getTest().getFeatures());
                labelsTrain.add(split.getTrain().getLabels());
                labelsTest.add(split.getTest().getLabels());
            }


            //Training of the model:
            int nEpochs1 = 15;
            for (int epoch = 0; epoch < nEpochs1; epoch++) {
                for (INDArray data : featuresTrain) {
                    net.fit(data, data);

                }
                System.out.println("Epoch " + epoch + " complete");
            }


            System.out.println(net.summary());


            //Layer reduction and new DataSet after features reduction
            MultiLayerNetwork redNet = new TransferLearning.Builder(net)
                .removeLayersFromOutput(2)
                .setFeatureExtractor(1)
                .build();
            System.out.println(redNet.summary());

            //Reshaping labels and new inputs creation
            iter.reset();
            INDArray features = Nd4j.create(0, 64);
            INDArray int_labels = Nd4j.create(0, 1).castTo(DataType.INT64);
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                List<INDArray> results = redNet.feedForward(ds.getFeatures(), false);
                features = Nd4j.concat(0, features, results.get(2));
                INDArray temp = Nd4j.argMax(ds.getLabels(), 1).reshape(Nd4j.argMax(ds.getLabels(), 1).shape()[0], 1);
                int_labels = Nd4j.concat(0, int_labels, temp);
            }

            DataSet newData = new DataSet(features, int_labels);
            File f = new File("prova2.csv");
            f.createNewFile();
            FileSplit fs = new FileSplit(f);
            NumberOfRecordsPartitioner nrp = new NumberOfRecordsPartitioner();
            CSVArrayRecordWriter recordWriter = new CSVArrayRecordWriter();
            recordWriter.initialize(fs, nrp);

            INDArray temp = Nd4j.concat(1, features, int_labels.castTo(DataType.FLOAT));
            for (int i = 0; i < temp.shape()[0]; i++) {
                recordWriter.write(RecordConverter.toRecord(temp.getRow(i)));
            }

            System.out.println("End");
        } else {


            RecordReader recordReader2 = new CSVRecordReader(0, ',');
            File f2 = new File("prova2.csv");
            recordReader2.initialize(new FileSplit(f2));
            DataSetIterator iter2 = new RecordReaderDataSetIterator.Builder(recordReader2, 100)
                .classification(64, 5)
                .build();


            List<INDArray> featuresTrain2 = new ArrayList<>();
            List<INDArray> featuresTest2 = new ArrayList<>();
            List<INDArray> labelsTrain2 = new ArrayList<>();
            List<INDArray> labelsTest2 = new ArrayList<>();

            while (iter2.hasNext()) {

                DataSet ds = iter2.next();
                SplitTestAndTrain split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
                featuresTrain2.add(split.getTrain().getFeatures());
                featuresTest2.add(split.getTest().getFeatures());
                labelsTrain2.add(split.getTrain().getLabels());
                labelsTest2.add(split.getTest().getLabels());
            }

            System.out.println("End2");




        //Inizializing classifier
        MultiLayerConfiguration classifierConfig = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(new AdaGrad(0.05))
            .activation(Activation.TANH)
            .l2(0.0001)
            .list()
            .layer(new DenseLayer.Builder().nIn(64).nOut(32)
                .build())
            .layer(new DenseLayer.Builder().nIn(32).nOut(16)
                .build())
            .layer(new DenseLayer.Builder().nIn(16).nOut(8)
                .build())
            .layer(new OutputLayer.Builder().nIn(8).nOut(5)
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .build()).validateOutputLayerConfig(false)
            .build();

        MultiLayerNetwork classifier = new MultiLayerNetwork(classifierConfig);
        classifier.setListeners(Collections.singletonList(new ScoreIterationListener(100)));

        //Reshaping labels

            List<INDArray> int_labelsTraining = new ArrayList<>();
            for (int i=0; i<labelsTrain2.size(); i++) {
                INDArray int_labelsTrain = Nd4j.create(0, 1).castTo(DataType.INT64);
                int_labelsTrain = Nd4j.argMax(labelsTrain2.get(i), 1).reshape(Nd4j.argMax(labelsTrain2.get(i), 1).shape()[0], 1);
                INDArrayIndex index = new SpecifiedIndex();
                index.init(0,int_labelsTrain.size(0));


               for (int j=0; j < int_labelsTrain.size(0); j++){
                   if (int_labelsTrain.get(index).equals(0) == false) {
                       int_labelsTrain.put(j,1,1);
                   }
                }
                int_labelsTraining.add(int_labelsTrain);
            }

            List<INDArray> int_labelsTesting = new ArrayList<>();
            for (int i=0; i<labelsTest2.size(); i++) {
                INDArray int_labelsTest = Nd4j.create(0, 1).castTo(DataType.INT64);
                int_labelsTest = Nd4j.argMax(labelsTest2.get(i), 1).reshape(Nd4j.argMax(labelsTest2.get(i), 1).shape()[0], 1);
                int_labelsTesting.add(int_labelsTest);
            }



            int nEpochs2 = 1;

            for (int epoch=0; epoch< nEpochs2; epoch++){
                for(int i=0; i<featuresTrain2.size(); i++){
            classifier.fit( featuresTrain2.get(i) , int_labelsTraining.get(i) );
            }
                System.out.println("Epoch "+epoch+" finished");
        }

            for (int i=0; i<featuresTest2.size(); i++) {
                Evaluation eval = new Evaluation(5);
                INDArray output = classifier.output(featuresTest2.get(i));
                eval.eval(int_labelsTesting.get(i),output );
                log.info(eval.stats());
            }



        }
    }
}











