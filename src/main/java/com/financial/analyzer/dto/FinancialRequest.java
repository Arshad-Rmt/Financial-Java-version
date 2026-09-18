package com.financial.analyzer.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import jakarta.validation.constraints.NotNull;

/**
 * Request DTO containing the 9 financial features required for financial distress prediction.
 * Supports standard camelCase as well as original Flask/dataset header names via @JsonAlias.
 */
public class FinancialRequest {

    @NotNull(message = "roaBeforeInterestAndDepreciationBeforeInterest is required")
    @JsonAlias({"ROA(C) before interest and depreciation before interest", "roa", "roa_before_interest"})
    private Double roaBeforeInterestAndDepreciationBeforeInterest;

    @NotNull(message = "debtRatio is required")
    @JsonAlias({"Debt ratio %", "debt_ratio", "debtRatioPercent"})
    private Double debtRatio;

    @NotNull(message = "netIncomeFlag is required")
    @JsonAlias({"Net Income Flag", "net_income_flag"})
    private Double netIncomeFlag;

    @NotNull(message = "currentRatio is required")
    @JsonAlias({"Current Ratio", "current_ratio"})
    private Double currentRatio;

    @NotNull(message = "operatingGrossMargin is required")
    @JsonAlias({"Operating Gross Margin", "operating_gross_margin", "operatingMargin"})
    private Double operatingGrossMargin;

    @NotNull(message = "interestCoverageRatio is required")
    @JsonAlias({"Interest Coverage Ratio (Interest expense to EBIT)", "interest_coverage_ratio", "interestCoverage"})
    private Double interestCoverageRatio;

    @NotNull(message = "equityToLiability is required")
    @JsonAlias({"Equity to Liability", "equity_to_liability"})
    private Double equityToLiability;

    @NotNull(message = "netIncomeToTotalAssets is required")
    @JsonAlias({"Net Income to Total Assets", "net_income_to_total_assets", "netIncomeToAssets"})
    private Double netIncomeToTotalAssets;

    @NotNull(message = "cashFlowPerShare is required")
    @JsonAlias({"Cash Flow Per Share", "cash_flow_per_share"})
    private Double cashFlowPerShare;

    public FinancialRequest() {
    }

    public FinancialRequest(Double roaBeforeInterestAndDepreciationBeforeInterest,
                            Double debtRatio,
                            Double netIncomeFlag,
                            Double currentRatio,
                            Double operatingGrossMargin,
                            Double interestCoverageRatio,
                            Double equityToLiability,
                            Double netIncomeToTotalAssets,
                            Double cashFlowPerShare) {
        this.roaBeforeInterestAndDepreciationBeforeInterest = roaBeforeInterestAndDepreciationBeforeInterest;
        this.debtRatio = debtRatio;
        this.netIncomeFlag = netIncomeFlag;
        this.currentRatio = currentRatio;
        this.operatingGrossMargin = operatingGrossMargin;
        this.interestCoverageRatio = interestCoverageRatio;
        this.equityToLiability = equityToLiability;
        this.netIncomeToTotalAssets = netIncomeToTotalAssets;
        this.cashFlowPerShare = cashFlowPerShare;
    }

    /**
     * Converts request features into a 1x9 float array in the EXACT order expected by the ONNX models:
     * 1. ROA(C) before interest and depreciation before interest
     * 2. Debt ratio %
     * 3. Net Income Flag
     * 4. Current Ratio
     * 5. Operating Gross Margin
     * 6. Interest Coverage Ratio (Interest expense to EBIT)
     * 7. Equity to Liability
     * 8. Net Income to Total Assets
     * 9. Cash Flow Per Share
     */
    public float[] toFeatureArray() {
        return new float[]{
                roaBeforeInterestAndDepreciationBeforeInterest.floatValue(),
                debtRatio.floatValue(),
                netIncomeFlag.floatValue(),
                currentRatio.floatValue(),
                operatingGrossMargin.floatValue(),
                interestCoverageRatio.floatValue(),
                equityToLiability.floatValue(),
                netIncomeToTotalAssets.floatValue(),
                cashFlowPerShare.floatValue()
        };
    }

    public Double getRoaBeforeInterestAndDepreciationBeforeInterest() {
        return roaBeforeInterestAndDepreciationBeforeInterest;
    }

    public void setRoaBeforeInterestAndDepreciationBeforeInterest(Double roaBeforeInterestAndDepreciationBeforeInterest) {
        this.roaBeforeInterestAndDepreciationBeforeInterest = roaBeforeInterestAndDepreciationBeforeInterest;
    }

    public Double getDebtRatio() {
        return debtRatio;
    }

    public void setDebtRatio(Double debtRatio) {
        this.debtRatio = debtRatio;
    }

    public Double getNetIncomeFlag() {
        return netIncomeFlag;
    }

    public void setNetIncomeFlag(Double netIncomeFlag) {
        this.netIncomeFlag = netIncomeFlag;
    }

    public Double getCurrentRatio() {
        return currentRatio;
    }

    public void setCurrentRatio(Double currentRatio) {
        this.currentRatio = currentRatio;
    }

    public Double getOperatingGrossMargin() {
        return operatingGrossMargin;
    }

    public void setOperatingGrossMargin(Double operatingGrossMargin) {
        this.operatingGrossMargin = operatingGrossMargin;
    }

    public Double getInterestCoverageRatio() {
        return interestCoverageRatio;
    }

    public void setInterestCoverageRatio(Double interestCoverageRatio) {
        this.interestCoverageRatio = interestCoverageRatio;
    }

    public Double getEquityToLiability() {
        return equityToLiability;
    }

    public void setEquityToLiability(Double equityToLiability) {
        this.equityToLiability = equityToLiability;
    }

    public Double getNetIncomeToTotalAssets() {
        return netIncomeToTotalAssets;
    }

    public void setNetIncomeToTotalAssets(Double netIncomeToTotalAssets) {
        this.netIncomeToTotalAssets = netIncomeToTotalAssets;
    }

    public Double getCashFlowPerShare() {
        return cashFlowPerShare;
    }

    public void setCashFlowPerShare(Double cashFlowPerShare) {
        this.cashFlowPerShare = cashFlowPerShare;
    }
}
