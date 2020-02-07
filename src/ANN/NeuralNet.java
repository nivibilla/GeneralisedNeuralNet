package ANN;

import trainset.TrainSet;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNet{

    private double[][]output;
    private double[][][]weights;
    private double[][]bias;
    private double[][]error;
    private double[][]output_derivative;

    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;

    public ArrayList<String[]> data;


    public NeuralNet(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];
        this.error = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];
        this.data = new ArrayList<>();

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i],0.3,0.7);
            this.error[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];
            if (i>0){
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i],NETWORK_LAYER_SIZES[i-1],-0.3,0.5);
            }
        }
    }

    public double[] calculate(double... input){
        if (input.length != INPUT_SIZE) return null;
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron <NETWORK_LAYER_SIZES[layer] ; neuron++) {
                double sum = bias[layer][neuron];

                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++) {
                    sum += output[layer-1][prevNeuron]*weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = sigmoid(sum);
                output_derivative[layer][neuron] = output[layer][neuron]*(1-output[layer][neuron]);

            }
            
        }

        return output[NETWORK_SIZE-1];
    }

    public void train(double[] input,double[] target, double eta){
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE);
        calculate(input);
        backpropogate(target);
        updateWeights(eta);
    }

    public void train(TrainSet set, int loops,int batch_size){
        for (int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            for (int b = 0; b < batch_size; b++) {
                this.train(batch.getInput(b),batch.getOutput(b), 0.3);
            }
        }
    }

    public void backpropogate(double[] target){
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; neuron++) {
            error[NETWORK_SIZE-1][neuron] = (output[NETWORK_SIZE-1][neuron] - target[neuron])*output_derivative[NETWORK_SIZE-1][neuron];
        }
        for (int layer = NETWORK_SIZE - 2; layer > 0 ; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron++) {
                    sum += weights[layer + 1][nextNeuron][neuron]*error[layer + 1][nextNeuron];
                }
                this.error[layer][neuron] = sum * output_derivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta){
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++) {
                    double delta = -eta*output[layer-1][prevNeuron]*error[layer][neuron];
                    weights[layer][neuron][prevNeuron] += delta;
                }
                double delta = -eta*error[layer][neuron];
                bias[layer][neuron] += delta;
            }
        }
    }

    public void saveWeights() throws IOException {
        ObjectOutputStream oos = null;
        FileOutputStream fout = null;
        try{
            fout = new FileOutputStream("weights.ser", true);
            oos = new ObjectOutputStream(fout);
            oos.writeObject(weights);
        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if(oos != null){
                oos.close();
            }
        }
    }
    
    public void loadWeights() throws IOException {
        ObjectInputStream ois = null;
        try {
            FileInputStream streamIn = new FileInputStream("weights.ser");
            ois = new ObjectInputStream(streamIn);
            weights = (double[][][]) ois.readObject();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if(ois != null){
                ois.close();
            }
        }
    }

    public void parseData(String filename){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String row;
            while((row = reader.readLine()) != null){
                data.add(row.split(","));
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.out.println("Invalid Filename, try including '.txt' or '.csv'");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private double sigmoid(double x){
        return 1d/(1 + Math.exp(-x));
    }

    public static void main(String[] args) throws IOException {
        NeuralNet net = new NeuralNet(4,3,3,2);
//        net.parseData("EURUSD.csv");
//        for(String[] x : net.data){
//            System.out.println(Arrays.toString(x));
//            int i = Integer.parseInt(x[5]);
//            if(x[5] == "0"){
//
//            }
//        }

        TrainSet set = new TrainSet(4,2);
        set.addData(new double[]{0.1,0.2,0.3,0.4}, new double[]{1,0});
        set.addData(new double[]{0.4,0.3,0.2,0.1}, new double[]{0,1});
        set.addData(new double[]{0.1,0.2,0.2,0.1}, new double[]{0,0});
        set.addData(new double[]{0.4,0.4,0.4,0.4}, new double[]{1,1});
        set.addData(new double[]{0.2,0.2,0.2,0.2}, new double[]{1,1});
        set.addData(new double[]{0.3,0.2,0.1,0}, new double[]{0,1});
        set.addData(new double[]{99,20,1,0.4}, new double[]{0,1});
        set.addData(new double[]{1,2,3,4}, new double[]{1,0});
        set.addData(new double[]{1,10,40,40}, new double[]{1,0});
        set.addData(new double[]{5,5,5,5}, new double[]{1,1});

        net.train(set,1000,set.size());


        for (int i = 0; i < set.size(); i++) {
            System.out.println(Arrays.toString(set.getOutput(i)));
            System.out.println(Arrays.toString(net.calculate(set.getInput(i))));
            System.out.println();
        }

        System.out.println("testtesttesttesttesttesttesttesttesttesttesttesttest");
        double[] input = new double[]{100,100,100,100};
        System.out.println(Arrays.toString(input));
        System.out.println(Arrays.toString(net.calculate(input)));



    }

}