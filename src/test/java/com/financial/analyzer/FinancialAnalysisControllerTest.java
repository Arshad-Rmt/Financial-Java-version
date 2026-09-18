package com.financial.analyzer;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
class FinancialAnalysisControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void testAnalyzeEndpointWithValidInput() throws Exception {
        String validJson = """
                {
                  "roaBeforeInterestAndDepreciationBeforeInterest": -0.25,
                  "debtRatio": 0.94,
                  "netIncomeFlag": 0,
                  "currentRatio": 0.45,
                  "operatingGrossMargin": 0.06,
                  "interestCoverageRatio": -1.0,
                  "equityToLiability": 0.18,
                  "netIncomeToTotalAssets": -0.11,
                  "cashFlowPerShare": -3.2
                }
                """;

        mockMvc.perform(post("/api/financial/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(validJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prediction", is(1)))
                .andExpect(jsonPath("$.probability", closeTo(0.695941, 0.001)))
                .andExpect(jsonPath("$.randomForestProbability", closeTo(0.380000, 0.001)))
                .andExpect(jsonPath("$.xgboostProbability", closeTo(0.932933, 0.001)))
                .andExpect(jsonPath("$.lightgbmProbability", closeTo(0.616919, 0.001)))
                .andExpect(jsonPath("$.weakPoints", hasSize(9)))
                .andExpect(jsonPath("$.weakPoints[0]", is("High Debt Ratio")))
                .andExpect(jsonPath("$.weakPoints[1]", is("Low Current Ratio")))
                .andExpect(jsonPath("$.weakPoints[2]", is("Low ROA")))
                .andExpect(jsonPath("$.weakPoints[3]", is("Low Operating Margin")))
                .andExpect(jsonPath("$.weakPoints[4]", is("Poor Interest Coverage")))
                .andExpect(jsonPath("$.weakPoints[5]", is("Low Equity to Liability")))
                .andExpect(jsonPath("$.weakPoints[6]", is("Low Net Income to Total Assets")))
                .andExpect(jsonPath("$.weakPoints[7]", is("Negative Cash Flow Per Share")))
                .andExpect(jsonPath("$.weakPoints[8]", is("Negative Net Income")))
                .andExpect(jsonPath("$.suggestions", hasSize(9)))
                .andExpect(jsonPath("$.suggestions[0]", is("Reduce debt ratio below 0.60")))
                .andExpect(jsonPath("$.suggestions[8]", is("Maintain consistent positive net income")));
    }

    @Test
    void testAnalyzeEndpointWithOriginalFlaskAliases() throws Exception {
        String aliasJson = """
                {
                  "ROA(C) before interest and depreciation before interest": -0.25,
                  "Debt ratio %": 0.94,
                  "Net Income Flag": 0,
                  "Current Ratio": 0.45,
                  "Operating Gross Margin": 0.06,
                  "Interest Coverage Ratio (Interest expense to EBIT)": -1.0,
                  "Equity to Liability": 0.18,
                  "Net Income to Total Assets": -0.11,
                  "Cash Flow Per Share": -3.2
                }
                """;

        mockMvc.perform(post("/api/financial/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(aliasJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prediction", is(1)))
                .andExpect(jsonPath("$.probability", closeTo(0.695941, 0.001)));
    }

    @Test
    void testAnalyzeEndpointWithMissingFieldReturns400() throws Exception {
        // Missing "debtRatio"
        String missingFieldJson = """
                {
                  "roaBeforeInterestAndDepreciationBeforeInterest": -0.25,
                  "netIncomeFlag": 0,
                  "currentRatio": 0.45,
                  "operatingGrossMargin": 0.06,
                  "interestCoverageRatio": -1.0,
                  "equityToLiability": 0.18,
                  "netIncomeToTotalAssets": -0.11,
                  "cashFlowPerShare": -3.2
                }
                """;

        mockMvc.perform(post("/api/financial/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(missingFieldJson))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status", is(400)))
                .andExpect(jsonPath("$.error", is("Validation Failed")))
                .andExpect(jsonPath("$.message", containsString("debtRatio is required")));
    }

    @Test
    void testAnalyzeEndpointWithMalformedJsonReturns400() throws Exception {
        String malformedJson = "{ invalid-json }";

        mockMvc.perform(post("/api/financial/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(malformedJson))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status", is(400)))
                .andExpect(jsonPath("$.error", is("Malformed JSON")));
    }

    @Test
    void testAnalyzeEndpointWithHealthyInputReturnsHealthyDiagnostics() throws Exception {
        String healthyJson = """
                {
                  "roaBeforeInterestAndDepreciationBeforeInterest": 0.15,
                  "debtRatio": 0.30,
                  "netIncomeFlag": 1,
                  "currentRatio": 2.50,
                  "operatingGrossMargin": 0.35,
                  "interestCoverageRatio": 5.00,
                  "equityToLiability": 2.00,
                  "netIncomeToTotalAssets": 0.10,
                  "cashFlowPerShare": 1.50
                }
                """;

        mockMvc.perform(post("/api/financial/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(healthyJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.weakPoints", hasSize(1)))
                .andExpect(jsonPath("$.weakPoints[0]", is("None")))
                .andExpect(jsonPath("$.suggestions", hasSize(1)))
                .andExpect(jsonPath("$.suggestions[0]", is("All financial indicators are within healthy ranges.")));
    }
}
