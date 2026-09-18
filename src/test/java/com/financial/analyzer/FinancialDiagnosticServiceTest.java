package com.financial.analyzer;

import com.financial.analyzer.dto.FinancialRequest;
import com.financial.analyzer.service.FinancialDiagnosticService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class FinancialDiagnosticServiceTest {

    private FinancialDiagnosticService diagnosticService;

    @BeforeEach
    void setUp() {
        diagnosticService = new FinancialDiagnosticService();
    }

    @Test
    @DisplayName("TEST 1: Manual Financial Input triggers all 9 diagnostic rules in exact order")
    void testManualFinancialInputTriggersAllRules() {
        FinancialRequest request = new FinancialRequest(
                -0.25, // roa < 0.05 -> Low ROA
                0.94,  // debtRatio > 0.70 -> High Debt Ratio
                0.0,   // netIncomeFlag == 0 -> Negative Net Income
                0.45,  // currentRatio < 1.50 -> Low Current Ratio
                0.06,  // operatingGrossMargin < 0.15 -> Low Operating Margin
                -1.0,  // interestCoverageRatio < 1.50 -> Poor Interest Coverage
                0.18,  // equityToLiability < 1.00 -> Low Equity to Liability
                -0.11, // netIncomeToTotalAssets < 0.03 -> Low Net Income to Total Assets
                -3.2   // cashFlowPerShare < 0 -> Negative Cash Flow Per Share
        );

        FinancialDiagnosticService.DiagnosticResult result = diagnosticService.evaluate(request);

        List<String> expectedWeakPoints = List.of(
                "High Debt Ratio",
                "Low Current Ratio",
                "Low ROA",
                "Low Operating Margin",
                "Poor Interest Coverage",
                "Low Equity to Liability",
                "Low Net Income to Total Assets",
                "Negative Cash Flow Per Share",
                "Negative Net Income"
        );

        List<String> expectedSuggestions = List.of(
                "Reduce debt ratio below 0.60",
                "Improve current ratio above 1.50",
                "Improve ROA above 0.05",
                "Improve operating margin above 0.15",
                "Improve interest coverage above 1.50",
                "Improve equity to liability above 1.00",
                "Improve net income to total assets above 0.03",
                "Maintain positive cash flow per share",
                "Maintain consistent positive net income"
        );

        assertEquals(expectedWeakPoints, result.weakPoints(), "Weak points must match expected in exact order");
        assertEquals(expectedSuggestions, result.suggestions(), "Suggestions must match expected in exact order");
    }

    @Test
    @DisplayName("TEST 2: Healthy diagnostic input triggers no rules and returns healthy defaults")
    void testHealthyDiagnosticInput() {
        FinancialRequest healthyRequest = new FinancialRequest(
                0.15, // roa >= 0.05
                0.30, // debtRatio <= 0.70
                1.0,  // netIncomeFlag != 0
                2.50, // currentRatio >= 1.50
                0.35, // operatingGrossMargin >= 0.15
                5.00, // interestCoverageRatio >= 1.50
                2.00, // equityToLiability >= 1.00
                0.10, // netIncomeToTotalAssets >= 0.03
                1.50  // cashFlowPerShare >= 0
        );

        FinancialDiagnosticService.DiagnosticResult result = diagnosticService.evaluate(healthyRequest);

        assertEquals(List.of("None"), result.weakPoints());
        assertEquals(List.of("All financial indicators are within healthy ranges."), result.suggestions());
    }

    @Test
    @DisplayName("TEST 3: Boundary values exactly on thresholds do NOT trigger diagnostic rules")
    void testBoundaryValuesDoNotTriggerRules() {
        FinancialRequest boundaryRequest = new FinancialRequest(
                0.05, // roa == 0.05 (threshold < 0.05) -> NO trigger
                0.70, // debtRatio == 0.70 (threshold > 0.70) -> NO trigger
                1.0,  // netIncomeFlag == 1 (threshold == 0) -> NO trigger
                1.50, // currentRatio == 1.50 (threshold < 1.50) -> NO trigger
                0.15, // operatingGrossMargin == 0.15 (threshold < 0.15) -> NO trigger
                1.50, // interestCoverageRatio == 1.50 (threshold < 1.50) -> NO trigger
                1.00, // equityToLiability == 1.00 (threshold < 1.00) -> NO trigger
                0.03, // netIncomeToTotalAssets == 0.03 (threshold < 0.03) -> NO trigger
                0.0   // cashFlowPerShare == 0 (threshold < 0) -> NO trigger
        );

        FinancialDiagnosticService.DiagnosticResult result = diagnosticService.evaluate(boundaryRequest);

        assertEquals(List.of("None"), result.weakPoints(),
                "Boundary values exactly on thresholds must not trigger any weak point");
        assertEquals(List.of("All financial indicators are within healthy ranges."), result.suggestions(),
                "Boundary values must return the healthy fallback suggestion");
    }

    @Test
    @DisplayName("Verify individual boundary triggering just past threshold")
    void testIndividualRuleTriggering() {
        // Debt ratio just above 0.70
        FinancialRequest reqDebt = createBaseHealthy();
        reqDebt.setDebtRatio(0.7001);
        FinancialDiagnosticService.DiagnosticResult resDebt = diagnosticService.evaluate(reqDebt);
        assertEquals(List.of("High Debt Ratio"), resDebt.weakPoints());
        assertEquals(List.of("Reduce debt ratio below 0.60"), resDebt.suggestions());

        // Current ratio just below 1.50
        FinancialRequest reqCurrent = createBaseHealthy();
        reqCurrent.setCurrentRatio(1.4999);
        FinancialDiagnosticService.DiagnosticResult resCurrent = diagnosticService.evaluate(reqCurrent);
        assertEquals(List.of("Low Current Ratio"), resCurrent.weakPoints());

        // Cash flow per share negative
        FinancialRequest reqCash = createBaseHealthy();
        reqCash.setCashFlowPerShare(-0.001);
        FinancialDiagnosticService.DiagnosticResult resCash = diagnosticService.evaluate(reqCash);
        assertEquals(List.of("Negative Cash Flow Per Share"), resCash.weakPoints());

        // Net income flag zero
        FinancialRequest reqFlag = createBaseHealthy();
        reqFlag.setNetIncomeFlag(0.0);
        FinancialDiagnosticService.DiagnosticResult resFlag = diagnosticService.evaluate(reqFlag);
        assertEquals(List.of("Negative Net Income"), resFlag.weakPoints());
    }

    @Test
    @DisplayName("Null request throws IllegalArgumentException")
    void testNullRequestThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> diagnosticService.evaluate(null));
    }

    private FinancialRequest createBaseHealthy() {
        return new FinancialRequest(
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
    }
}
