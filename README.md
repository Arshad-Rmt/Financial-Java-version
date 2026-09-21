# Financial Health Analyzer

Financial Health Analyzer is a Spring Boot web application that evaluates nine company financial indicators and estimates financial distress risk. It runs three exported machine-learning models with ONNX Runtime, combines their class-1 probabilities into a weighted ensemble prediction, and returns deterministic diagnostic weak points and suggestions alongside the model output.

## Live Demo

[Open the deployed Financial Health Analyzer](https://financial-java-version.onrender.com/)

## Key Features

- Browser-based form for the nine financial inputs.
- REST API for programmatic analysis.
- Random Forest, XGBoost, and LightGBM inference through ONNX Runtime.
- Weighted soft-voting ensemble with weights `[1, 2, 2]`.
- Distress classification threshold of `0.55`.
- Individual model probabilities and weighted ensemble probability in the response.
- Deterministic weak-point rules and recommendations based on financial thresholds.
- Financial metrics chart comparing submitted values with diagnostic reference thresholds.
- Client-side required-field validation and server-side request validation.
- Dark and light UI themes.
- Startup model inspection and an internal inference check for the bundled ONNX models.

## Tech Stack

- Java 21
- Spring Boot 3.3.0
- Spring Web
- Spring Boot Validation
- Microsoft ONNX Runtime Java API 1.20.0
- Maven
- HTML, CSS, and vanilla JavaScript for the web interface
- Python scripts using scikit-learn, XGBoost, LightGBM, imbalanced-learn, joblib, skl2onnx, onnxmltools, ONNX, and ONNX Runtime for training, conversion, and validation
- Docker with Maven and Eclipse Temurin Java 21 images

## Architecture and Workflow

```text
Browser UI
    |
    | POST /api/financial/analyze
    v
FinancialAnalysisController
    |
    v
FinancialAnalysisService
    |-- converts the request to the required 9-value feature array
    |-- calls OnnxModelService
    |-- applies weighted soft voting and the 0.55 threshold
    `-- calls FinancialDiagnosticService
            |
            v
       FinancialResponse JSON
```

At application startup, `OnnxModelService` loads the three ONNX files from the classpath, checks the expected `float_input` input and `probabilities` output, and runs an internal sample inference. During a request, the three already-loaded sessions produce class-1 probabilities. `FinancialAnalysisService` calculates:

```text
ensemble probability = (1 * RF + 2 * XGBoost + 2 * LightGBM) / 5
prediction = 1 when ensemble probability >= 0.55, otherwise 0
```

The diagnostic service independently checks the submitted ratios and returns ordered weak points and suggestions. The static JavaScript client renders the prediction, probability gauge, diagnostics, recommendations, and chart.

## Machine-Learning Approach

The Python training script selects the nine features below, creates a stratified 80/20 train/test split, applies SMOTE to the training data, and trains three classifiers:

- Random Forest
- XGBoost
- LightGBM

The training script builds a soft-voting ensemble with exact weights `[1, 2, 2]` in the order Random Forest, XGBoost, and LightGBM. The classification threshold is `0.55`. The conversion script extracts the individual estimators from `model.pkl` and exports them as separate ONNX models for the Java application.

### Why ONNX is used for Java deployment

ONNX provides a model exchange format that lets the trained Python estimators be executed by the Java service without requiring the Python training environment at runtime. Microsoft ONNX Runtime supplies the Java inference API, while the checked-in model files can be loaded from Spring Boot classpath resources. This keeps the deployed application as a Java service and avoids retraining or loading Python model objects in production.

The repository contains the exported ONNX models under `onnx-models/` and packaged application resources under `src/main/resources/models/`. The Python scripts that retrain or reconvert the models expect a local `model.pkl`; that file is not included in the repository tree.

## Input Features

The API requires these nine features, in this exact order:

1. `roaBeforeInterestAndDepreciationBeforeInterest` - ROA before interest and depreciation before interest
2. `debtRatio` - Debt ratio
3. `netIncomeFlag` - Net income flag
4. `currentRatio` - Current ratio
5. `operatingGrossMargin` - Operating gross margin
6. `interestCoverageRatio` - Interest coverage ratio (interest expense to EBIT)
7. `equityToLiability` - Equity to liability
8. `netIncomeToTotalAssets` - Net income to total assets
9. `cashFlowPerShare` - Cash flow per share

All nine values are required JSON numbers. The DTO also accepts aliases matching the original dataset column names and several shorter aliases.

## REST API

### `POST /api/financial/analyze`

Content type: `application/json`

Example request:

```json
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
```

Example response for the repository's manual test input:

```json
{
  "prediction": 1,
  "probability": 0.695941,
  "randomForestProbability": 0.38,
  "xgboostProbability": 0.932933,
  "lightgbmProbability": 0.616919,
  "weakPoints": [
    "High Debt Ratio",
    "Low Current Ratio",
    "Low ROA",
    "Low Operating Margin",
    "Poor Interest Coverage",
    "Low Equity to Liability",
    "Low Net Income to Total Assets",
    "Negative Cash Flow Per Share",
    "Negative Net Income"
  ],
  "suggestions": [
    "Reduce debt ratio below 0.60",
    "Improve current ratio above 1.50",
    "Improve ROA above 0.05",
    "Improve operating margin above 0.15",
    "Improve interest coverage above 1.50",
    "Improve equity to liability above 1.00",
    "Improve net income to total assets above 0.03",
    "Maintain positive cash flow per share",
    "Maintain consistent positive net income"
  ]
}
```

Validation failures, malformed JSON, and other handled errors are returned as structured error responses by the global exception handler.

## Project Structure

```text
.
├── data.csv/                         Dataset files
├── onnx-models/                      Exported ONNX models and metadata
├── python-files/                     Training, conversion, and validation scripts
├── src/
│   ├── main/java/com/financial/analyzer/
│   │   ├── controller/               REST controller
│   │   ├── dto/                      Request, response, and error DTOs
│   │   ├── exception/                Global exception handling
│   │   └── service/                  Inference, orchestration, and diagnostics
│   ├── main/resources/
│   │   ├── models/                   ONNX models packaged in the application
│   │   └── static/                   HTML, CSS, and JavaScript UI
│   └── test/java/                    Spring Boot, API, service, and rule tests
├── Dockerfile                        Multi-stage Docker build
└── pom.xml                           Maven configuration
```

## Local Setup and Run

### Prerequisites

- JDK 21
- Maven 3.9 or later recommended

From the repository root, run the application with:

```bash
mvn spring-boot:run
```

The default server port is `8080`. Open [http://localhost:8080](http://localhost:8080).

To create the packaged Spring Boot application:

```bash
mvn clean package
```

The Java application uses the ONNX models already packaged in `src/main/resources/models`. The Python training and conversion scripts are optional and require their own Python dependencies plus the absent local `model.pkl` input.

## Testing

Run the Java test suite with:

```bash
mvn test
```

The tests cover application context startup, ONNX model initialization and inference, valid and invalid API requests, ensemble probabilities, diagnostic rules, healthy inputs, and threshold boundaries.

The Python validation script can compare Python and ONNX inference when a compatible local `model.pkl` is available:

```bash
python python-files/validate_onnx.py
```

## Docker and Render Deployment

The included `Dockerfile` uses a multi-stage build:

1. Maven with Eclipse Temurin Java 21 downloads dependencies and packages the application.
2. An Eclipse Temurin Java 21 JRE image runs the resulting JAR.

The container exposes port `8080`. Its entrypoint uses the Render-provided `PORT` environment variable when present and otherwise defaults to `8080`:

```bash
docker build -t financial-health-analyzer .
docker run --rm -p 8080:8080 financial-health-analyzer
```

The deployed application is available at the [live demo](https://financial-java-version.onrender.com/).

## Screenshots

Add screenshots of the implemented interface here for repository and project-review use:

![Financial Health Analyzer input form](docs/screenshots/input-form.png)

![Financial Health Analyzer analysis result](docs/screenshots/analysis-result.png)

The image paths above are placeholders; no screenshot files are included in the repository tree.

## Limitations and Disclaimer

- The model outputs are estimates based on the bundled exported models and the nine submitted indicators; they are not a guarantee of a company's future financial condition.
- This project is for demonstration and analytical support, not financial, investment, accounting, legal, or credit advice.
- The application does not expose a user authentication or authorization layer in the checked-in source.
- The API accepts numeric inputs but does not define domain-range constraints for each financial ratio beyond required-field validation.
- The Python retraining and ONNX conversion workflow requires a local `model.pkl`, which is not checked in.
- No performance, accuracy, or production business outcome should be inferred from the example response. The repository tests verify implementation behavior for selected inputs, not general model quality.

## Future Improvements

- Add a documented reproducible Python environment and model-training artifact workflow.
- Add model evaluation reports, calibration checks, and dataset/version tracking.
- Add broader input validation and clearer API documentation, such as OpenAPI/Swagger.
- Add authentication, authorization, rate limiting, and production observability for a public deployment.
- Add automated UI and API integration tests in CI.
- Add persisted analysis history and exportable reports.
- Add monitoring for model drift and a controlled model versioning process.