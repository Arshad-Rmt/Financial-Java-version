package com.financial.analyzer.service;

import com.financial.analyzer.dto.FinancialRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Service responsible for evaluating deterministic financial diagnostic rules.
 * Identifies weak points and suggestions based on financial ratios in exact sequence.
 */
@Service
public class FinancialDiagnosticService {

    private static final Logger logger = LoggerFactory.getLogger(FinancialDiagnosticService.class);

    public record DiagnosticResult(List<String> weakPoints, List<String> suggestions) {}

    /**
     * Evaluates diagnostic rules against the provided FinancialRequest.
     *
     * @param request FinancialRequest containing financial metrics.
     * @return DiagnosticResult containing ordered lists of weakPoints and suggestions.
     */
    public DiagnosticResult evaluate(FinancialRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("Financial request cannot be null.");
        }

        List<String> weakPoints = new ArrayList<>();
        List<String> suggestions = new ArrayList<>();

        // 1. Debt Ratio: debtRatio > 0.70
        if (request.getDebtRatio() != null && request.getDebtRatio() > 0.70) {
            weakPoints.add("High Debt Ratio");
            suggestions.add("Reduce debt ratio below 0.60");
        }

        // 2. Current Ratio: currentRatio < 1.50
        if (request.getCurrentRatio() != null && request.getCurrentRatio() < 1.50) {
            weakPoints.add("Low Current Ratio");
            suggestions.add("Improve current ratio above 1.50");
        }

        // 3. ROA: roaBeforeInterestAndDepreciationBeforeInterest < 0.05
        if (request.getRoaBeforeInterestAndDepreciationBeforeInterest() != null
                && request.getRoaBeforeInterestAndDepreciationBeforeInterest() < 0.05) {
            weakPoints.add("Low ROA");
            suggestions.add("Improve ROA above 0.05");
        }

        // 4. Operating Gross Margin: operatingGrossMargin < 0.15
        if (request.getOperatingGrossMargin() != null && request.getOperatingGrossMargin() < 0.15) {
            weakPoints.add("Low Operating Margin");
            suggestions.add("Improve operating margin above 0.15");
        }

        // 5. Interest Coverage Ratio: interestCoverageRatio < 1.50
        if (request.getInterestCoverageRatio() != null && request.getInterestCoverageRatio() < 1.50) {
            weakPoints.add("Poor Interest Coverage");
            suggestions.add("Improve interest coverage above 1.50");
        }

        // 6. Equity to Liability: equityToLiability < 1.00
        if (request.getEquityToLiability() != null && request.getEquityToLiability() < 1.00) {
            weakPoints.add("Low Equity to Liability");
            suggestions.add("Improve equity to liability above 1.00");
        }

        // 7. Net Income to Total Assets: netIncomeToTotalAssets < 0.03
        if (request.getNetIncomeToTotalAssets() != null && request.getNetIncomeToTotalAssets() < 0.03) {
            weakPoints.add("Low Net Income to Total Assets");
            suggestions.add("Improve net income to total assets above 0.03");
        }

        // 8. Cash Flow Per Share: cashFlowPerShare < 0
        if (request.getCashFlowPerShare() != null && request.getCashFlowPerShare() < 0.0) {
            weakPoints.add("Negative Cash Flow Per Share");
            suggestions.add("Maintain positive cash flow per share");
        }

        // 9. Net Income Flag: netIncomeFlag == 0
        if (request.getNetIncomeFlag() != null && (request.getNetIncomeFlag() == 0.0 || request.getNetIncomeFlag().intValue() == 0)) {
            weakPoints.add("Negative Net Income");
            suggestions.add("Maintain consistent positive net income");
        }

        // Healthy Case: If NONE of the above conditions are triggered
        if (weakPoints.isEmpty()) {
            weakPoints.add("None");
            suggestions.add("All financial indicators are within healthy ranges.");
        }

        logger.info("Diagnostics evaluated - Weak points count: {}", weakPoints.size());

        return new DiagnosticResult(
                Collections.unmodifiableList(weakPoints),
                Collections.unmodifiableList(suggestions)
        );
    }
}
