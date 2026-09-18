package com.financial.analyzer.dto;

import java.util.List;

/**
 * Response DTO returning the ensemble prediction, overall weighted probability,
 * individual component probabilities from Random Forest, XGBoost, and LightGBM,
 * and deterministic financial weak points and suggestions.
 */
public class FinancialResponse {

    private int prediction;
    private double probability;
    private double randomForestProbability;
    private double xgboostProbability;
    private double lightgbmProbability;
    private List<String> weakPoints;
    private List<String> suggestions;

    public FinancialResponse() {
    }

    public FinancialResponse(int prediction,
                             double probability,
                             double randomForestProbability,
                             double xgboostProbability,
                             double lightgbmProbability) {
        this(prediction, probability, randomForestProbability, xgboostProbability, lightgbmProbability, null, null);
    }

    public FinancialResponse(int prediction,
                             double probability,
                             double randomForestProbability,
                             double xgboostProbability,
                             double lightgbmProbability,
                             List<String> weakPoints,
                             List<String> suggestions) {
        this.prediction = prediction;
        this.probability = probability;
        this.randomForestProbability = randomForestProbability;
        this.xgboostProbability = xgboostProbability;
        this.lightgbmProbability = lightgbmProbability;
        this.weakPoints = weakPoints;
        this.suggestions = suggestions;
    }

    public int getPrediction() {
        return prediction;
    }

    public void setPrediction(int prediction) {
        this.prediction = prediction;
    }

    public double getProbability() {
        return probability;
    }

    public void setProbability(double probability) {
        this.probability = probability;
    }

    public double getRandomForestProbability() {
        return randomForestProbability;
    }

    public void setRandomForestProbability(double randomForestProbability) {
        this.randomForestProbability = randomForestProbability;
    }

    public double getXgboostProbability() {
        return xgboostProbability;
    }

    public void setXgboostProbability(double xgboostProbability) {
        this.xgboostProbability = xgboostProbability;
    }

    public double getLightgbmProbability() {
        return lightgbmProbability;
    }

    public void setLightgbmProbability(double lightgbmProbability) {
        this.lightgbmProbability = lightgbmProbability;
    }

    public List<String> getWeakPoints() {
        return weakPoints;
    }

    public void setWeakPoints(List<String> weakPoints) {
        this.weakPoints = weakPoints;
    }

    public List<String> getSuggestions() {
        return suggestions;
    }

    public void setSuggestions(List<String> suggestions) {
        this.suggestions = suggestions;
    }
}
