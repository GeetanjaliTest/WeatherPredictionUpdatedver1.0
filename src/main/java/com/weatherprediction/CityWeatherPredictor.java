package com.weatherprediction;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;


import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class CityWeatherPredictor {
    private static final Map<String, INDArray> weatherData = new HashMap<>();
    private static MultiLayerNetwork model;

    static {
        try {
            loadWeatherData("weather-dataset.csv");
            loadModel("src/main/resources/weather-model.zip");  // Fixed: Ensure model is in correct path
        } catch (IOException e) {
            System.err.println("Error initializing application: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void loadWeatherData(String filePath) throws IOException {
        InputStream inputStream = new ClassPathResource(filePath).getInputStream();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 2) { // Ensure at least city + one feature
                    String city = parts[0].trim().toLowerCase();
                    double[] features = new double[parts.length - 1];
                    for (int i = 1; i < parts.length; i++) {
                        try {
                            features[i - 1] = Double.parseDouble(parts[i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Invalid feature data for city: " + city);
                        }
                    }
                    weatherData.put(city, Nd4j.create(features));
                }
            }
        }
    }

    private static void loadModel(String modelPath) {
        try {
            File modelFile = new ClassPathResource(modelPath).getFile();
            if (!modelFile.exists() || modelFile.length() == 0) {
                System.err.println("Model file is missing or empty: " + modelPath);
                return;
            }
            model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            System.out.println("Model successfully loaded.");
        } catch (IOException e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static String predictWeather(String city) {
        if (model == null) {
            return "Error: Model is not loaded. Please check 'weather-model.zip'.";
        }
        if (StringUtils.isBlank(city)) {
            return "Invalid city name.";
        }
        INDArray features = weatherData.get(city.toLowerCase());
        if (features == null) {
            return "City data not found!";
        }
        INDArray prediction = model.output(features);
        return "Predicted Weather Values: " + prediction.toString();
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter a city name to predict its weather: ");
        String city = scanner.nextLine().trim();
        String weather = predictWeather(city);
        System.out.println("Predicted weather for '" + city + "': " + weather);
    }
}
