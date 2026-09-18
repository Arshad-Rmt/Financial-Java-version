package com.financial.analyzer;

import com.financial.analyzer.service.OnnxModelService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class FinancialHealthAnalyzerApplicationTests {

    @Autowired
    private OnnxModelService onnxModelService;

    @Test
    void contextLoads() {
        assertNotNull(onnxModelService, "OnnxModelService should be injected");
        assertNotNull(onnxModelService.getEnvironment(), "OrtEnvironment should be initialized");
        assertNotNull(onnxModelService.getSessionRf(), "Random Forest session should be initialized");
        assertNotNull(onnxModelService.getSessionXgb(), "XGBoost session should be initialized");
        assertNotNull(onnxModelService.getSessionLgbm(), "LightGBM session should be initialized");

        var details = onnxModelService.getInspectionDetails();
        assertEquals(3, details.size(), "Should have inspection details for 3 models");

        assertTrue(details.get("Random Forest").isVerificationPassed(), "Random Forest verification should pass");
        assertTrue(details.get("XGBoost").isVerificationPassed(), "XGBoost verification should pass");
        assertTrue(details.get("LightGBM").isVerificationPassed(), "LightGBM verification should pass");
    }
}

