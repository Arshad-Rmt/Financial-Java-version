package com.financial.analyzer.service;

import ai.onnxruntime.OrtException;
import com.financial.analyzer.dto.FinancialRequest;
import com.financial.analyzer.dto.FinancialResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Business service responsible for orchestrating financial distress analysis:
 * extracts the 9 ordered features, invokes the ONNX models, calculates the weighted soft voting,
 * and determines the classification decision based on threshold 0.55.
 */
@Service
public class FinancialAnalysisService {

    private static final Logger logger = LoggerFactory.getLogger(FinancialAnalysisService.class);

    // Original ensemble weights: RF=1, XGB=2, LGBM=2, Total=5
    public static final double WEIGHT_RANDOM_FOREST = 1.0;
    public static final double WEIGHT_XGBOOST = 2.0;
    public static final double WEIGHT_LIGHTGBM = 2.0;
    public static final double TOTAL_WEIGHT = 5.0;

    // Classification threshold: 0.55
    public static final double PREDICTION_THRESHOLD = 0.55;

    private final OnnxModelService onnxModelService;
    private final FinancialDiagnosticService financialDiagnosticService;

    public FinancialAnalysisService(OnnxModelService onnxModelService,
                                   FinancialDiagnosticService financialDiagnosticService) {
        this.onnxModelService = onnxModelService;
        this.financialDiagnosticService = financialDiagnosticService;
    }

    /**
     * Analyzes financial distress risk using the ensemble of 3 ONNX models
     * and evaluates deterministic financial diagnostic rules.
     *
     * @param request FinancialRequest containing the 9 financial features.
     * @return FinancialResponse with ensemble prediction, probabilities, weak points, and suggestions.
     * @throws OrtException if an error occurs during ONNX inference.
     */
    public FinancialResponse analyze(FinancialRequest request) throws OrtException {
        if (request == null) {
            throw new IllegalArgumentException("Financial request cannot be null.");
        }

        float[] features = request.toFeatureArray();

        OnnxModelService.ModelProbabilities probs = onnxModelService.predictAll(features);

        double rfProb = probs.rfProb();
        double xgbProb = probs.xgbProb();
        double lgbmProb = probs.lgbmProb();

        // Weighted soft voting: (1 * RF + 2 * XGB + 2 * LGBM) / 5
        double weightedProb = (WEIGHT_RANDOM_FOREST * rfProb
                + WEIGHT_XGBOOST * xgbProb
                + WEIGHT_LIGHTGBM * lgbmProb) / TOTAL_WEIGHT;

        // Custom threshold 0.55
        int prediction = (weightedProb >= PREDICTION_THRESHOLD) ? 1 : 0;

        logger.info("Inference completed - RF: {}, XGB: {}, LGBM: {}, Ensemble Prob: {}, Prediction: {}",
                round(rfProb, 6), round(xgbProb, 6), round(lgbmProb, 6), round(weightedProb, 6), prediction);

        // Evaluate financial diagnostics (deterministic business logic)
        FinancialDiagnosticService.DiagnosticResult diagnostics = financialDiagnosticService.evaluate(request);

        return new FinancialResponse(
                prediction,
                round(weightedProb, 6),
                round(rfProb, 6),
                round(xgbProb, 6),
                round(lgbmProb, 6),
                diagnostics.weakPoints(),
                diagnostics.suggestions()
        );
    }

    private double round(double value, int places) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return value;
        }
        return BigDecimal.valueOf(value)
                .setScale(places, RoundingMode.HALF_UP)
                .doubleValue();
    }
}
