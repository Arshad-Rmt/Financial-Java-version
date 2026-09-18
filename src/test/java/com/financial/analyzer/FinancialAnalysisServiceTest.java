package com.financial.analyzer;

import ai.onnxruntime.OrtException;
import com.financial.analyzer.dto.FinancialRequest;
import com.financial.analyzer.dto.FinancialResponse;
import com.financial.analyzer.service.FinancialAnalysisService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class FinancialAnalysisServiceTest {

    @Autowired
    private FinancialAnalysisService financialAnalysisService;

    @Test
    void testManualTestSampleInference() throws OrtException {
        // Manual validation input from 4_hybrid_model.py:
        // [-0.25, 0.94, 0, 0.45, 0.06, -1.0, 0.18, -0.11, -3.2]
        FinancialRequest request = new FinancialRequest(
                -0.25,
                0.94,
                0.0,
                0.45,
                0.06,
                -1.0,
                0.18,
                -0.11,
                -3.2
        );

        FinancialResponse response = financialAnalysisService.analyze(request);

        assertNotNull(response, "Response should not be null");

        // Verify individual component probabilities
        assertEquals(0.380000, response.getRandomForestProbability(), 0.001,
                "Random Forest probability should be approximately 0.380000");
        assertEquals(0.932933, response.getXgboostProbability(), 0.001,
                "XGBoost probability should be approximately 0.932933");
        assertEquals(0.616919, response.getLightgbmProbability(), 0.001,
                "LightGBM probability should be approximately 0.616919");

        // Verify weighted soft voting probability: (1*0.38 + 2*0.932933 + 2*0.616919) / 5 ≈ 0.695941
        assertEquals(0.695941, response.getProbability(), 0.001,
                "Ensemble probability should be approximately 0.695941");

        // Verify prediction: 0.695941 >= 0.55 threshold -> 1 (Bankrupt/Distressed)
        assertEquals(1, response.getPrediction(),
                "Prediction should be 1 for probability >= 0.55");

        // Verify Phase 4 diagnostics
        assertNotNull(response.getWeakPoints(), "Weak points should not be null");
        assertEquals(9, response.getWeakPoints().size(), "Should have all 9 weak points");
        assertEquals("High Debt Ratio", response.getWeakPoints().get(0));
        assertEquals("Low Current Ratio", response.getWeakPoints().get(1));
        assertEquals("Low ROA", response.getWeakPoints().get(2));
        assertEquals("Low Operating Margin", response.getWeakPoints().get(3));
        assertEquals("Poor Interest Coverage", response.getWeakPoints().get(4));
        assertEquals("Low Equity to Liability", response.getWeakPoints().get(5));
        assertEquals("Low Net Income to Total Assets", response.getWeakPoints().get(6));
        assertEquals("Negative Cash Flow Per Share", response.getWeakPoints().get(7));
        assertEquals("Negative Net Income", response.getWeakPoints().get(8));

        assertNotNull(response.getSuggestions(), "Suggestions should not be null");
        assertEquals(9, response.getSuggestions().size(), "Should have all 9 suggestions");
        assertEquals("Reduce debt ratio below 0.60", response.getSuggestions().get(0));
        assertEquals("Improve current ratio above 1.50", response.getSuggestions().get(1));
        assertEquals("Improve ROA above 0.05", response.getSuggestions().get(2));
        assertEquals("Improve operating margin above 0.15", response.getSuggestions().get(3));
        assertEquals("Improve interest coverage above 1.50", response.getSuggestions().get(4));
        assertEquals("Improve equity to liability above 1.00", response.getSuggestions().get(5));
        assertEquals("Improve net income to total assets above 0.03", response.getSuggestions().get(6));
        assertEquals("Maintain positive cash flow per share", response.getSuggestions().get(7));
        assertEquals("Maintain consistent positive net income", response.getSuggestions().get(8));
    }

    @Test
    void testHealthyDiagnosticInput() throws OrtException {
        FinancialRequest healthyRequest = new FinancialRequest(
                0.15,
                0.30,
                1.0,
                2.50,
                0.35,
                5.00,
                2.00,
                0.10,
                1.50
        );

        FinancialResponse response = financialAnalysisService.analyze(healthyRequest);
        assertNotNull(response);
        assertEquals(java.util.List.of("None"), response.getWeakPoints());
        assertEquals(java.util.List.of("All financial indicators are within healthy ranges."), response.getSuggestions());
    }

    @Test
    void testBoundaryDiagnosticInput() throws OrtException {
        FinancialRequest boundaryRequest = new FinancialRequest(
                0.05,
                0.70,
                1.0,
                1.50,
                0.15,
                1.50,
                1.00,
                0.03,
                0.0
        );

        FinancialResponse response = financialAnalysisService.analyze(boundaryRequest);
        assertNotNull(response);
        assertEquals(java.util.List.of("None"), response.getWeakPoints());
        assertEquals(java.util.List.of("All financial indicators are within healthy ranges."), response.getSuggestions());
    }

    @Test
    void testNullRequestThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> financialAnalysisService.analyze(null));
    }
}
