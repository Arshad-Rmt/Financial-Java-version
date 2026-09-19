# Financial Health Analyzer

Financial Health Analyzer is a Spring Boot application that evaluates a company's financial metrics with a hybrid ONNX model ensemble. It provides a browser-based interface for submitting nine financial features and returns a distress prediction, probability details, diagnostic weak points, and suggestions.

## Key Features

- Web UI for entering the nine financial metrics used by the models.
- REST endpoint for financial analysis.
- ONNX Runtime inference with Random Forest, XGBoost, and LightGBM models.
- Weighted soft voting across the three model probabilities.
- Deterministic financial diagnostic rules with ordered weak points and suggestions.
- Startup inspection and inference checks for all bundled ONNX models.
- Light and dark UI themes, input validation, result visualization, and a financial metrics chart.

## Tech Stack

- Java 21
- Spring Boot 3.3.0
- Spring Web
- Spring Boot Validation
- Microsoft ONNX Runtime Java API 1.20.0
- Maven
- HTML, CSS, and vanilla JavaScript for the static web UI
- Python scripts for model training, ONNX conversion, and ONNX validation

## Architecture and Workflow

1. The Spring Boot application serves the static interface from `src/main/resources/static`.
2. The browser validates the nine submitted values and sends them as JSON to `POST /api/financial/analyze`.
3. `FinancialAnalysisController` passes the request to `FinancialAnalysisService`.
4. `FinancialRequest` converts the values into the exact nine-feature order expected by the ONNX models.
5. `OnnxModelService` loads the three models from the classpath at startup, verifies their tensor names, and reuses their ONNX Runtime sessions for inference.
6. `FinancialAnalysisService` combines the class-1 probabilities using the weights `[1, 2, 2]`, applies the `0.55` classification threshold, and invokes `FinancialDiagnosticService`.
7. The JSON response contains the ensemble result, component probabilities, weak points, and suggestions. The browser renders the result and chart.

## ML and ONNX Approach

The Python workflow trains a soft-voting ensemble containing:

- Random Forest
- XGBoost
- LightGBM

The ensemble uses weights of `1`, `2`, and `2`, respectively, with a classification threshold of `0.55`. `convert_to_onnx.py` exports the individual estimators as `random_forest.onnx`, `xgboost.onnx`, and `lightgbm.onnx`, and writes the feature and model metadata to `model_metadata.json`.

The Java application loads the three ONNX files from `src/main/resources/models`. Each model expects the input tensor `float_input` and exposes class probabilities through `probabilities`. `validate_onnx.py` compares the Python and ONNX inference paths on the manual sample and, when the required Python model file is available, on a stratified test split.

The checked-in Python scripts reference `model.pkl`, but that file is not present in the repository tree. The already-exported ONNX models used by the Java application are included under both `onnx-models` and the application resources directory.

## API

### `POST /api/financial/analyze`

Content type: `application/json`

The request accepts these nine required camelCase fields. The DTO also supports aliases matching the original dataset headers.

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

The response contains `prediction`, `probability`, `randomForestProbability`, `xgboostProbability`, `lightgbmProbability`, `weakPoints`, and `suggestions`.

## Project Structure

```text
.
├── data.csv/                 Dataset files
├── onnx-models/              Exported ONNX models and metadata
├── python-files/             Training, conversion, and validation scripts
├── src/main/java/            Spring Boot application, API, DTOs, and services
├── src/main/resources/
│   ├── models/               ONNX models packaged with the application
│   └── static/                Browser UI assets
├── src/test/java/             Spring Boot, controller, service, and diagnostic tests
└── pom.xml                   Maven build and dependency configuration
```

## Setup and Run

### Prerequisites

- JDK 21
- Maven

### Run the application

From the repository root:

```bash
mvn spring-boot:run
```

The application is configured to listen on port `8080`. Open [http://localhost:8080](http://localhost:8080) in a browser.

### Build the package

```bash
mvn clean package
```

The Spring Boot Maven plugin creates the application package under `target`.

### Optional Python model workflow

The Python scripts are separate from the Java runtime application. `4_hybrid_model.py` trains and saves `model.pkl`; `convert_to_onnx.py` reads that file and writes the ONNX exports; `validate_onnx.py` compares the Python and ONNX predictions. These scripts require the Python dependencies imported by the scripts and a matching `model.pkl` file.

## Testing

Run the automated Java test suite with:

```bash
mvn test
```

The tests cover application context startup, ONNX model initialization, controller validation and responses, ensemble inference, and diagnostic rules.

## Screenshots and Demo

Screenshots and a live demo link can be added here when available.

<!-- Screenshot placeholder: add an image of the running Financial Health Analyzer UI. -->

<!-- Demo placeholder: add a hosted demo or short walkthrough link when available. -->

## Repository Metadata

Suggested GitHub repository description:

> Spring Boot financial distress analyzer using a weighted Random Forest, XGBoost, and LightGBM ONNX ensemble.

Suggested GitHub topics:

`java` `spring-boot` `machine-learning` `onnx` `onnx-runtime` `financial-analysis` `xgboost` `lightgbm` `random-forest` `maven`