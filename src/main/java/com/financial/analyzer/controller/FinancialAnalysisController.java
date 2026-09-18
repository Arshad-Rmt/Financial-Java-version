package com.financial.analyzer.controller;

import ai.onnxruntime.OrtException;
import com.financial.analyzer.dto.FinancialRequest;
import com.financial.analyzer.dto.FinancialResponse;
import com.financial.analyzer.service.FinancialAnalysisService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * REST controller exposing endpoints for financial distress prediction and analysis.
 */
@RestController
@RequestMapping("/api/financial")
@CrossOrigin(origins = "*")
public class FinancialAnalysisController {

    private static final Logger logger = LoggerFactory.getLogger(FinancialAnalysisController.class);

    private final FinancialAnalysisService financialAnalysisService;

    public FinancialAnalysisController(FinancialAnalysisService financialAnalysisService) {
        this.financialAnalysisService = financialAnalysisService;
    }

    /**
     * Analyzes company financial metrics and predicts financial distress using ONNX hybrid ensemble.
     *
     * @param request FinancialRequest containing the 9 required financial metrics.
     * @return ResponseEntity with FinancialResponse JSON.
     * @throws OrtException if model inference fails.
     */
    @PostMapping("/analyze")
    public ResponseEntity<FinancialResponse> analyze(@Valid @RequestBody FinancialRequest request) throws OrtException {
        logger.info("Received financial analysis request");
        FinancialResponse response = financialAnalysisService.analyze(request);
        return ResponseEntity.ok(response);
    }
}
