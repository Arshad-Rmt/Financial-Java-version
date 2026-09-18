package com.financial.analyzer.service;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Dedicated service responsible for initializing ONNX Runtime,
 * loading the three trained ONNX models (Random Forest, XGBoost, LightGBM),
 * verifying their tensor input/output specifications, and providing thread-safe inference.
 */
@Service
public class OnnxModelService {

    private static final Logger logger = LoggerFactory.getLogger(OnnxModelService.class);

    private static final String MODEL_PATH_RF = "models/random_forest.onnx";
    private static final String MODEL_PATH_XGB = "models/xgboost.onnx";
    private static final String MODEL_PATH_LGBM = "models/lightgbm.onnx";
    private static final String INPUT_TENSOR_NAME = "float_input";
    private static final String OUTPUT_TENSOR_NAME = "probabilities";

    private OrtEnvironment environment;
    private OrtSession.SessionOptions sessionOptions;

    private OrtSession sessionRf;
    private OrtSession sessionXgb;
    private OrtSession sessionLgbm;

    private final Map<String, ModelInspectionDetails> inspectionDetails = new LinkedHashMap<>();

    /**
     * Immutable container for the class 1 probabilities predicted by the three individual models.
     */
    public record ModelProbabilities(double rfProb, double xgbProb, double lgbmProb) {}

    public static class ModelInspectionDetails {
        private final String modelName;
        private final Map<String, String> inputs = new LinkedHashMap<>();
        private final Map<String, String> outputs = new LinkedHashMap<>();
        private boolean verificationPassed;
        private String verificationMessage;

        public ModelInspectionDetails(String modelName) {
            this.modelName = modelName;
        }

        public String getModelName() {
            return modelName;
        }

        public Map<String, String> getInputs() {
            return inputs;
        }

        public Map<String, String> getOutputs() {
            return outputs;
        }

        public boolean isVerificationPassed() {
            return verificationPassed;
        }

        public void setVerificationPassed(boolean verificationPassed) {
            this.verificationPassed = verificationPassed;
        }

        public String getVerificationMessage() {
            return verificationMessage;
        }

        public void setVerificationMessage(String verificationMessage) {
            this.verificationMessage = verificationMessage;
        }
    }

    @PostConstruct
    public void initialize() {
        logger.info("=================================================================");
        logger.info("Initializing ONNX Runtime Environment and Loading Models...");
        logger.info("=================================================================");

        try {
            this.environment = OrtEnvironment.getEnvironment("FinancialHealthAnalyzer");
            this.sessionOptions = new OrtSession.SessionOptions();
            this.sessionOptions.setIntraOpNumThreads(1);
            this.sessionOptions.setInterOpNumThreads(1);

            // Load each model from classpath resources
            this.sessionRf = loadSession("Random Forest", MODEL_PATH_RF);
            this.sessionXgb = loadSession("XGBoost", MODEL_PATH_XGB);
            this.sessionLgbm = loadSession("LightGBM", MODEL_PATH_LGBM);

            logger.info("All 3 ONNX sessions created successfully.");

            // Startup Verification
            verifyModel("Random Forest", this.sessionRf);
            verifyModel("XGBoost", this.sessionXgb);
            verifyModel("LightGBM", this.sessionLgbm);

            // Test execution with manual sample to ensure no native runtime crashes
            testInternalInference();

            logger.info("=================================================================");
            logger.info("ONNX Runtime Model Loading and Startup Verification: COMPLETE");
            logger.info("=================================================================");

        } catch (Exception e) {
            logger.error("Failed to initialize ONNX Runtime or load models: {}", e.getMessage(), e);
            throw new IllegalStateException("ONNX Model initialization failed", e);
        }
    }

    private OrtSession loadSession(String modelName, String resourcePath) throws IOException, OrtException {
        ClassPathResource resource = new ClassPathResource(resourcePath);
        if (!resource.exists()) {
            throw new IllegalArgumentException("Model resource not found: " + resourcePath);
        }

        logger.info("Loading model '{}' from classpath resource: {}", modelName, resourcePath);
        try (InputStream inputStream = resource.getInputStream()) {
            byte[] modelBytes = inputStream.readAllBytes();
            logger.info("Model '{}' loaded ({} bytes). Creating OrtSession...", modelName, modelBytes.length);
            return environment.createSession(modelBytes, sessionOptions);
        }
    }

    private void verifyModel(String modelName, OrtSession session) throws OrtException {
        ModelInspectionDetails details = new ModelInspectionDetails(modelName);

        logger.info("--- Inspecting Model: {} ---", modelName);

        // Discover and inspect Inputs
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        logger.info("  Input count: {}", inputInfo.size());
        for (Map.Entry<String, NodeInfo> entry : inputInfo.entrySet()) {
            String name = entry.getKey();
            NodeInfo info = entry.getValue();
            String desc = formatNodeInfo(info);
            details.getInputs().put(name, desc);
            logger.info("  Input [{}]: {}", name, desc);
        }

        // Discover and inspect Outputs
        Map<String, NodeInfo> outputInfo = session.getOutputInfo();
        logger.info("  Output count: {}", outputInfo.size());
        for (Map.Entry<String, NodeInfo> entry : outputInfo.entrySet()) {
            String name = entry.getKey();
            NodeInfo info = entry.getValue();
            String desc = formatNodeInfo(info);
            details.getOutputs().put(name, desc);
            logger.info("  Output [{}]: {}", name, desc);
        }

        boolean hasExpectedInput = inputInfo.containsKey(INPUT_TENSOR_NAME);
        boolean hasExpectedOutput = outputInfo.containsKey(OUTPUT_TENSOR_NAME);

        if (hasExpectedInput && hasExpectedOutput) {
            details.setVerificationPassed(true);
            details.setVerificationMessage("OK: Contains expected input '" + INPUT_TENSOR_NAME + "' and output '" + OUTPUT_TENSOR_NAME + "'");
            logger.info("  [VERIFICATION SUCCESS] {}", details.getVerificationMessage());
        } else {
            details.setVerificationPassed(false);
            details.setVerificationMessage(String.format("Mismatch! hasExpectedInput(%s)=%b, hasExpectedOutput(%s)=%b",
                    INPUT_TENSOR_NAME, hasExpectedInput, OUTPUT_TENSOR_NAME, hasExpectedOutput));
            logger.warn("  [VERIFICATION WARNING] {}", details.getVerificationMessage());
        }

        this.inspectionDetails.put(modelName, details);
    }

    private String formatNodeInfo(NodeInfo info) {
        if (info.getInfo() instanceof TensorInfo tensorInfo) {
            return String.format("type=%s, shape=%s", tensorInfo.type, Arrays.toString(tensorInfo.getShape()));
        }
        return info.getInfo().toString();
    }

    private void testInternalInference() {
        logger.info("Performing internal mock inference test across all 3 models...");
        // Manual test sample from 4_hybrid_model.py
        float[] testSample = new float[]{-0.25f, 0.94f, 0.0f, 0.45f, 0.06f, -1.0f, 0.18f, -0.11f, -3.2f};

        try {
            ModelProbabilities probs = predictAll(testSample);
            logger.info("  Internal test RF prob class 1: {}", probs.rfProb());
            logger.info("  Internal test XGBoost prob class 1: {}", probs.xgbProb());
            logger.info("  Internal test LightGBM prob class 1: {}", probs.lgbmProb());
            logger.info("  Internal mock inference test PASSED on all 3 models.");
        } catch (Exception e) {
            logger.error("Internal mock inference test failed: {}", e.getMessage(), e);
            throw new IllegalStateException("Internal inference verification failed", e);
        }
    }

    /**
     * Executes inference across the 3 pre-loaded models with the given 9-feature input.
     * Thread-safe and reuses existing OrtSession instances without reloading.
     *
     * @param features float array of exactly 9 financial features in the required order.
     * @return ModelProbabilities containing class-1 probabilities for RF, XGB, and LGBM.
     * @throws OrtException if an error occurs during ONNX inference.
     */
    public ModelProbabilities predictAll(float[] features) throws OrtException {
        if (features == null || features.length != 9) {
            throw new IllegalArgumentException("Input features array must be non-null and have length 9.");
        }

        float[][] input2D = new float[][]{features};

        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, input2D)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(INPUT_TENSOR_NAME, tensor);

            double rfProb = extractClass1Probability(sessionRf.run(inputs));
            double xgbProb = extractClass1Probability(sessionXgb.run(inputs));
            double lgbmProb = extractClass1Probability(sessionLgbm.run(inputs));

            return new ModelProbabilities(rfProb, xgbProb, lgbmProb);
        }
    }

    private double extractClass1Probability(OrtSession.Result result) throws OrtException {
        try (result) {
            OnnxValue probVal = result.get(OUTPUT_TENSOR_NAME).orElse(result.get(1));
            float[][] probs = (float[][]) probVal.getValue();
            return probs[0][1];
        }
    }

    public OrtEnvironment getEnvironment() {
        return environment;
    }

    public OrtSession getSessionRf() {
        return sessionRf;
    }

    public OrtSession getSessionXgb() {
        return sessionXgb;
    }

    public OrtSession getSessionLgbm() {
        return sessionLgbm;
    }

    public Map<String, ModelInspectionDetails> getInspectionDetails() {
        return Collections.unmodifiableMap(inspectionDetails);
    }

    @PreDestroy
    public void close() {
        logger.info("Closing ONNX sessions and environment...");
        try {
            if (sessionRf != null) sessionRf.close();
            if (sessionXgb != null) sessionXgb.close();
            if (sessionLgbm != null) sessionLgbm.close();
            if (sessionOptions != null) sessionOptions.close();
            if (environment != null) environment.close();
            logger.info("ONNX resources successfully released.");
        } catch (Exception e) {
            logger.error("Error closing ONNX resources: {}", e.getMessage(), e);
        }
    }
}
