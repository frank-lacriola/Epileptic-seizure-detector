import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.deeplearning4j.core.evaluation.EvaluationTools;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class EpSAutoencoder2 {

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
            int nEpochs1 = 500;
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

            Schema schema2 = new Schema.Builder()
                    .addColumnsDouble("param_%d", 1, 64)
                    .addColumnDouble("class")
                    .build();

            TransformProcess transformProcess2 = new TransformProcess.Builder(schema2)
                    .conditionalReplaceValueTransform("class", new IntWritable(0), new DoubleColumnCondition("class", ConditionOp.LessOrEqual, 3))
                    .conditionalReplaceValueTransform("class", new IntWritable(1), new DoubleColumnCondition("class", ConditionOp.Equal, 4))
                    .build();

            //Passing transformation process to convert the csv file
            RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader2, transformProcess2);


            DataSetIterator iter2 = new RecordReaderDataSetIterator.Builder(transformProcessRecordReader, 100)
                    .classification(64, 2)
                    .build();


            INDArray features = Nd4j.zeros(0, 64);
            INDArray labels = Nd4j.zeros(0, 2);
            while (iter2.hasNext()) {
                DataSet ds = iter2.next();
                features = Nd4j.vstack(features, ds.getFeatures()); //concatenazione
                labels = Nd4j.vstack(labels, ds.getLabels());
            }
            int[] c0 = {1,0};
            int[] c1 = {0,1};
            long [] shape = {2};
// creiamo vettore per classi 0 e 1 di features e labels
            INDArray class0features = Nd4j.zeros(0, 64);
            INDArray class0labels = Nd4j.zeros(0, 2);

            INDArray class1features = Nd4j.zeros(0, 64);
            INDArray class1labels = Nd4j.zeros(0, 2);

            INDArray ClassBool0 =  labels.eq(Nd4j.create(c0, shape, DataType.FLOAT)).getColumn(0).castTo(DataType.DOUBLE);
            for (int i = 0; i < ClassBool0.shape()[0]; i++) {
                if (ClassBool0.getDouble(i) == 1) {
                    class0features = Nd4j.vstack(class0features, features.getRow(i).reshape(1,64));
                    class0labels = Nd4j.vstack(class0labels, labels.getRow(i).reshape(1,2));
                }
                else {
                    class1features = Nd4j.vstack(class1features, features.getRow(i).reshape(1,64));
                    class1labels = Nd4j.vstack(class1labels, labels.getRow(i).reshape(1,2));
                }
            }


            Nd4j.shuffle(class0features, r, 0);
            Nd4j.shuffle(class1features, r, 0);

            long[] dims = {class0features.shape()[0], class1features.shape()[0]};
            long [] shape2 = {2};
            int minNumber = Nd4j.create(dims, shape2, DataType.FLOAT).minNumber().intValue();

            INDArray finalFeatures = Nd4j.zeros(0, 64);
            INDArray finalLabels = Nd4j.zeros(0, 2);

            INDArray testFeatures = Nd4j.zeros(0, 64);
            INDArray testLabels = Nd4j.zeros(0, 2);

            for (int i = 0; i < minNumber; i++) {
                finalFeatures = Nd4j.vstack(finalFeatures, class0features.getRow(i).reshape(1,64));
                finalFeatures = Nd4j.vstack(finalFeatures, class1features.getRow(i).reshape(1,64));

                finalLabels = Nd4j.vstack(finalLabels, class0labels.getRow(i).reshape(1,2));
                finalLabels = Nd4j.vstack(finalLabels, class1labels.getRow(i).reshape(1,2));
            }

            if (class0features.shape()[0] > minNumber) {
                for (int i = minNumber; i < class0features.shape()[0]; i++)
                {
                    testFeatures = Nd4j.vstack(testFeatures, class0features.getRow(i).reshape(1,64));
                    testLabels = Nd4j.vstack(testLabels, class0labels.getRow(i).reshape(1,2));
                }
            }
            else {
                for (int i = minNumber; i < class1features.shape()[0]; i++)
                {
                    testFeatures = Nd4j.vstack(testFeatures, class1features.getRow(i).reshape(1,64));
                    testLabels = Nd4j.vstack(testLabels, class1labels.getRow(i).reshape(1,2));
                }
            }

            DataSet ds = new DataSet(finalFeatures, finalLabels);
            SplitTestAndTrain split = ds.splitTestAndTrain((int)(minNumber*0.8), r);  //80/20 split (from miniBatch = 100)
            testFeatures = Nd4j.vstack(testFeatures, split.getTest().getFeatures());
            testLabels = Nd4j.vstack(testLabels, split.getTest().getLabels());


            //Inizializing classifier
            MultiLayerConfiguration classifierConfig = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new AdaGrad(0.1))
                    .activation(Activation.TANH)
                    .l2(0.0001)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(64).nOut(32)
                            .build())
                    .layer(new DenseLayer.Builder().nIn(32).nOut(16)
                            .activation(Activation.SIGMOID)
                            .build())
                    .layer(new OutputLayer.Builder().nIn(16).nOut(2)
                            .activation(Activation.SIGMOID)
                            .lossFunction(LossFunctions.LossFunction.XENT)
                            .build())
                    .build();

            MultiLayerNetwork classifier = new MultiLayerNetwork(classifierConfig);
            classifier.init();
            classifier.setListeners(Collections.singletonList(new ScoreIterationListener(100)));



            int nEpochs2 = 500;

            for (int epoch=0; epoch< nEpochs2; epoch++){
                classifier.fit(split.getTrain());
                //System.out.println("Epoch "+epoch+" finished");
            }



            Evaluation eval = new Evaluation(2);
            INDArray output = classifier.output(testFeatures);
            eval.eval(testLabels, output );
            log.info(eval.stats());

            eval = new Evaluation(2);
            output = classifier.output(finalFeatures);
            eval.eval(finalLabels, output );


            ROC e2 = new ROC();
            output = classifier.output(testFeatures);
            e2.eval(testLabels, output);
            PrintWriter out = new PrintWriter("roccurve.html");
            out.print(EvaluationTools.rocChartToHtml(e2));

        }
    }
}