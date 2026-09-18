/**
 * FinInsight — Financial Health Analyzer
 * Frontend UI Controller: Theme Management, Chart Rendering, API Integration
 *
 * BACKEND INTEGRATION:
 * Calls the Spring Boot endpoint POST /api/financial/analyze (FinancialAnalysisController).
 * Request body matches FinancialRequest exactly (9 required Double fields, camelCase).
 * Response is deserialized from FinancialResponse:
 *   { prediction: int, probability: double, randomForestProbability, xgboostProbability,
 *     lightgbmProbability, weakPoints: string[], suggestions: string[] }
 * Only prediction, probability, weakPoints and suggestions are used/displayed —
 * the individual model probabilities are intentionally ignored in the UI.
 *
 * EMPTY-START BEHAVIOUR:
 * The form starts completely empty. Any `value="..."` left in the HTML is stripped
 * on load, example placeholders are applied, the prediction card stays in its
 * "Awaiting Analysis" state, and the chart shows only the neutral Healthy Threshold
 * reference — no Input Value series is drawn until an analysis succeeds.
 */

(function () {
  'use strict';

  const API_ENDPOINT = '/api/financial/analyze';

  // -------------------------------------------------------------
  // 1. Theme Management (Dark & Default/Light Theme)
  // -------------------------------------------------------------
  const THEME_STORAGE_KEY = 'fininsight_theme';
  const htmlElement = document.documentElement;
  const btnDark = document.getElementById('btn-dark-theme');
  const btnLight = document.getElementById('btn-light-theme');
  const themeToggle = document.getElementById('theme-toggle');

  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) || 'dark';
    setTheme(savedTheme);
  }

  function setTheme(theme) {
    htmlElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }

  if (btnDark) {
    btnDark.addEventListener('click', (e) => {
      e.stopPropagation();
      setTheme('dark');
    });
  }

  if (btnLight) {
    btnLight.addEventListener('click', (e) => {
      e.stopPropagation();
      setTheme('light');
    });
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = htmlElement.getAttribute('data-theme') || 'dark';
      setTheme(current === 'dark' ? 'light' : 'dark');
    });
  }

  // -------------------------------------------------------------
  // 2. Financial Metrics Chart Renderer
  // -------------------------------------------------------------
  // Metric order matches the form field order / FinancialRequest.toFeatureArray().

  const X_COORDS = [50, 105, 160, 215, 270, 325, 380, 435, 490];
  const Y_ZERO = 80;
  const Y_SCALE = 15; // 15px per unit (4 -> 20px, 0 -> 80px, -4 -> 140px)

  // "Healthy Threshold" — the same diagnostic baseline used to flag weak
  // points server-side. This is a minimum reference line, NOT a recommended,
  // ideal, or best-case value, and is rendered dashed/hollow to make that
  // distinction visually clear (see .chart-line-threshold in style.css).
  const HEALTHY_THRESHOLDS = [
    0.05,  // ROA Before Interest & Depreciation
    0.60,  // Debt Ratio
    1,     // Net Income Flag
    1.50,  // Current Ratio
    0.15,  // Operating Gross Margin
    1.50,  // Interest Coverage Ratio
    1.00,  // Equity to Liability
    0.03,  // Net Income to Total Assets
    0      // Cash Flow Per Share
  ];

  function valueToY(val) {
    const clamped = Math.max(-4, Math.min(4, val));
    return Y_ZERO - (clamped * Y_SCALE);
  }

  function generateSmoothPath(points) {
    if (points.length === 0) return '';
    if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;

    let path = `M ${points[0].x} ${points[0].y}`;
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = (i > 0) ? points[i - 1] : points[i];
      const p1 = points[i];
      const p2 = points[i + 1];
      const p3 = (i < points.length - 2) ? points[i + 2] : p2;

      const cp1x = p1.x + (p2.x - p0.x) / 5;
      const cp1y = p1.y + (p2.y - p0.y) / 5;
      const cp2x = p2.x - (p3.x - p1.x) / 5;
      const cp2y = p2.y - (p3.y - p1.y) / 5;

      path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`;
    }
    return path;
  }

  function plotDots(pointsGroup, points, dotClass, radius) {
    if (!pointsGroup) return;
    pointsGroup.innerHTML = '';
    points.forEach(p => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', p.x);
      circle.setAttribute('cy', p.y);
      circle.setAttribute('r', radius);
      circle.setAttribute('class', dotClass);
      pointsGroup.appendChild(circle);
    });
  }

  // Draws the fixed Healthy Threshold reference series. It carries no user data,
  // so it is safe to show before an analysis has run.
  function renderThresholdSeries() {
    const thresholdPathEl = document.getElementById('chart-threshold-path');
    const thresholdPointsGroup = document.getElementById('chart-threshold-points-group');
    if (!thresholdPathEl) return;

    const thresholdPoints = HEALTHY_THRESHOLDS.map((val, idx) => ({
      x: X_COORDS[idx],
      y: valueToY(val)
    }));
    thresholdPathEl.setAttribute('d', generateSmoothPath(thresholdPoints));
    plotDots(thresholdPointsGroup, thresholdPoints, 'chart-dot-threshold', '3');
  }

  // Neutral/empty chart state: no Input Value line, area or dots at all.
  // Used on first load and whenever a previous result is cleared (validation
  // error, API failure, or while a request is in flight).
  function clearChart() {
    const inputPathEl = document.getElementById('chart-input-path');
    const areaPathEl = document.getElementById('chart-area-path');
    const pointsGroup = document.getElementById('chart-points-group');

    if (inputPathEl) inputPathEl.setAttribute('d', '');
    if (areaPathEl) areaPathEl.setAttribute('d', '');
    if (pointsGroup) pointsGroup.innerHTML = '';

    renderThresholdSeries();
  }

  // Draws the Input Value series against the Healthy Threshold reference.
  // Only called once an analysis has completed successfully.
  function updateChart(values) {
    const inputPathEl = document.getElementById('chart-input-path');
    const areaPathEl = document.getElementById('chart-area-path');
    const pointsGroup = document.getElementById('chart-points-group');

    if (!inputPathEl || !pointsGroup) return;

    // --- Series 1: Input Value (from the 9 submitted form fields) ---
    const points = values.map((val, idx) => ({
      x: X_COORDS[idx],
      y: valueToY(val)
    }));

    const smoothLine = generateSmoothPath(points);
    inputPathEl.setAttribute('d', smoothLine);

    if (areaPathEl) {
      const areaD = `${smoothLine} L ${points[points.length - 1].x} ${Y_ZERO} L ${points[0].x} ${Y_ZERO} Z`;
      areaPathEl.setAttribute('d', areaD);
    }

    plotDots(pointsGroup, points, 'chart-dot', '3.5');

    // --- Series 2: Healthy Threshold (fixed diagnostic reference line) ---
    renderThresholdSeries();
  }

  // -------------------------------------------------------------
  // 3. Form Helpers
  // -------------------------------------------------------------
  // Field order matches FinancialRequest.toFeatureArray() / the chart's X axis.
  // `placeholder` is an illustrative example of the expected scale only — it is
  // never read, never submitted, and never used as a default value.
  const FIELD_META = [
    { id: 'input-roa',          label: 'ROA Before Interest & Depreciation', placeholder: 'e.g. 0.05' },
    { id: 'input-debt',         label: 'Debt Ratio',                         placeholder: 'e.g. 0.60' },
    { id: 'input-netflag',      label: 'Net Income Flag',                    placeholder: '0 or 1' },
    { id: 'input-current',      label: 'Current Ratio',                      placeholder: 'e.g. 1.50' },
    { id: 'input-opmargin',     label: 'Operating Gross Margin',             placeholder: 'e.g. 0.15' },
    { id: 'input-interestcov',  label: 'Interest Coverage Ratio',            placeholder: 'e.g. 1.50' },
    { id: 'input-equityliab',   label: 'Equity to Liability',                placeholder: 'e.g. 1.00' },
    { id: 'input-netassets',    label: 'Net Income to Total Assets',         placeholder: 'e.g. 0.03' },
    { id: 'input-cashflow',     label: 'Cash Flow Per Share',                placeholder: 'e.g. 1.20' }
  ];
  const FIELD_IDS = FIELD_META.map(f => f.id);

  // Starts the form completely empty and applies the example placeholders.
  // Any `value="..."` still present in the markup is stripped here, so the form
  // is empty on load and stays empty after a browser reload/restore.
  function initForm() {
    FIELD_META.forEach(({ id, placeholder }) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.removeAttribute('value');
      el.value = '';
      el.setAttribute('autocomplete', 'off');
      if (placeholder) el.setAttribute('placeholder', placeholder);
    });
  }

  // Lenient read of the current form state. Kept for the public API only —
  // it is not used to draw anything before an analysis has run.
  function getFormValues() {
    return FIELD_IDS.map(id => parseFloat(document.getElementById(id)?.value) || 0);
  }

  // Strict validation used before every /api/financial/analyze call.
  // FinancialRequest requires all 9 fields to be non-null numbers (@NotNull),
  // so empty or non-numeric inputs must be caught client-side rather than
  // silently coerced to 0.
  function validateInputs() {
    const values = {};
    const missingLabels = [];

    FIELD_META.forEach(({ id, label }) => {
      const el = document.getElementById(id);
      const raw = el ? el.value.trim() : '';
      const num = raw === '' ? NaN : Number(raw);
      if (raw === '' || Number.isNaN(num)) {
        missingLabels.push(label);
      } else {
        values[id] = num;
      }
    });

    return { values, missingLabels };
  }

  function buildPayload(values) {
    const payload = {};
    FIELD_META.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el && el.name) {
        payload[el.name] = values[id];
      }
    });
    return payload;
  }

  // -------------------------------------------------------------
  // 4. Prediction Result Rendering
  // -------------------------------------------------------------
  const GAUGE_CIRCUMFERENCE = 314.159;

  const statusLabelEl = document.getElementById('prediction-status-label');
  const statusDescEl = document.getElementById('prediction-status-desc');
  const iconBoxEl = document.getElementById('prediction-icon-box');
  const gaugeValueEl = document.getElementById('gauge-value');
  const gaugeProgressEl = document.getElementById('gauge-progress');
  const weakPointsListEl = document.getElementById('weak-points-list');
  const weakPointsCountEl = document.getElementById('weak-points-count');
  const recommendationsListEl = document.getElementById('recommendations-list');
  const recommendationsCountEl = document.getElementById('recommendations-count');
  const formErrorEl = document.getElementById('form-error');

  function normalizeProbability(rawProbability) {
    let prob = Number(rawProbability);
    if (Number.isNaN(prob)) prob = 0;
    if (prob <= 1) prob *= 100;
    return Math.max(0, Math.min(100, prob));
  }

  function resolveIsHighRisk(data, probabilityPercent) {
    const pred = data.prediction;
    if (typeof pred === 'boolean') return pred;
    if (typeof pred === 'number') return pred === 1;
    if (typeof pred === 'string') {
      const normalized = pred.trim().toLowerCase();
      if (['1', 'true', 'risk', 'high risk', 'distress', 'bankrupt', 'bankruptcy'].includes(normalized)) return true;
      if (['0', 'false', 'safe', 'low risk', 'no risk', 'healthy'].includes(normalized)) return false;
    }
    // Fallback: derive from probability if prediction field is missing/ambiguous
    return probabilityPercent >= 50;
  }

  function renderList(listEl, countEl, items, emptyMessage) {
    listEl.innerHTML = '';
    if (!items || items.length === 0) {
      const li = document.createElement('li');
      li.className = 'empty-item';
      li.textContent = emptyMessage;
      listEl.appendChild(li);
      countEl.textContent = '0';
      return;
    }
    items.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      listEl.appendChild(li);
    });
    countEl.textContent = String(items.length);
  }

  function renderPredictionResult(data) {
    const probabilityPercent = normalizeProbability(data.probability);
    const isHighRisk = resolveIsHighRisk(data, probabilityPercent);

    // Status heading / description / icon
    statusLabelEl.textContent = isHighRisk ? 'High Risk' : 'Low Risk';
    statusLabelEl.className = 'status-heading ' + (isHighRisk ? 'text-red' : 'text-green');
    statusDescEl.textContent = isHighRisk
      ? 'This company shows signs of financial distress.'
      : 'This company appears to be in stable financial health.';
    iconBoxEl.className = 'alert-icon-square ' + (isHighRisk ? 'status-danger' : 'status-safe');

    // Gauge
    const offset = GAUGE_CIRCUMFERENCE * (1 - probabilityPercent / 100);
    gaugeProgressEl.style.strokeDashoffset = String(offset);
    gaugeProgressEl.style.stroke = isHighRisk ? 'var(--red-accent)' : 'var(--green-accent)';
    gaugeValueEl.textContent = probabilityPercent.toFixed(2) + '%';

    // Weak points & recommendations (from existing API response fields).
    // randomForestProbability / xgboostProbability / lightgbmProbability are
    // deliberately not read or displayed.
    renderList(weakPointsListEl, weakPointsCountEl, data.weakPoints, 'No significant weak points identified.');
    renderList(recommendationsListEl, recommendationsCountEl, data.suggestions, 'No specific recommendations — financial health looks stable.');
  }

  // Neutral defaults — identical text/markup to the page's initial HTML state,
  // so "no result yet" always looks the same whether that's on first load,
  // after a validation error, or after a failed API call.
  const INITIAL_STATUS_LABEL = 'Awaiting Analysis';
  const INITIAL_STATUS_DESC = 'Fill in the financial details above and click "Analyze Company" to get a prediction.';
  const INITIAL_WEAK_POINTS_MESSAGE = 'Run an analysis to see identified weak points.';
  const INITIAL_RECOMMENDATIONS_MESSAGE = 'Run an analysis to see recommendations.';

  // Resets the prediction card, weak points, and recommendations to either:
  //  - 'initial': the neutral "no result" state (used on first load, and to
  //     clear/hide any previous prediction on validation or API errors), or
  //  - 'loading': a transient "Analyzing…" state shown while the request
  //     to /api/financial/analyze is in flight.
  function resetPredictionSection(mode) {
    const isLoading = mode === 'loading';

    statusLabelEl.textContent = isLoading ? 'Analyzing…' : INITIAL_STATUS_LABEL;
    statusLabelEl.className = 'status-heading';
    statusDescEl.textContent = isLoading
      ? 'Running the hybrid model on your inputs…'
      : INITIAL_STATUS_DESC;
    iconBoxEl.className = 'alert-icon-square' + (isLoading ? ' is-loading' : '');

    gaugeProgressEl.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
    gaugeProgressEl.style.stroke = 'var(--neutral-accent)';
    gaugeValueEl.textContent = isLoading ? '…' : '—';

    renderList(weakPointsListEl, weakPointsCountEl, null, isLoading ? 'Analyzing…' : INITIAL_WEAK_POINTS_MESSAGE);
    renderList(recommendationsListEl, recommendationsCountEl, null, isLoading ? 'Analyzing…' : INITIAL_RECOMMENDATIONS_MESSAGE);

    // No result means no Input Value series — the chart falls back to the
    // neutral threshold-only state in both modes.
    clearChart();
  }

  // -------------------------------------------------------------
  // 5. Analyze Action (calls existing Spring Boot API)
  // -------------------------------------------------------------
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnAnalyzeLabel = document.getElementById('btn-analyze-label');

  function setLoading(isLoading) {
    if (!btnAnalyze) return;
    btnAnalyze.disabled = isLoading;
    btnAnalyzeLabel.textContent = isLoading ? 'Analyzing…' : 'Analyze Company';
  }

  function showFormError(message) {
    if (!formErrorEl) return;
    if (!message) {
      formErrorEl.hidden = true;
      formErrorEl.textContent = '';
      return;
    }
    formErrorEl.hidden = false;
    formErrorEl.textContent = message;
  }

  // Attempts to extract a human-readable message from the backend's
  // GlobalExceptionHandler JSON error body (ErrorResponse: status/error/message/path).
  async function extractServerErrorMessage(response) {
    try {
      const body = await response.json();
      return body.message || body.error || null;
    } catch (_) {
      return null;
    }
  }

  async function handleAnalyze() {
    showFormError(null);

    const { values, missingLabels } = validateInputs();
    if (missingLabels.length > 0) {
      // Invalid input: clear/hide any previous prediction and chart series,
      // then show only the error.
      resetPredictionSection('initial');
      showFormError(
        missingLabels.length === FIELD_META.length
          ? 'Please fill in all 9 financial fields with numeric values.'
          : `Please enter valid numeric values for: ${missingLabels.join(', ')}.`
      );
      return;
    }

    // Clear the previous result and enter the loading/analyzing state.
    resetPredictionSection('loading');

    setLoading(true);
    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload(values))
      });

      if (!response.ok) {
        const serverMessage = await extractServerErrorMessage(response);
        throw new Error(serverMessage || ('Server responded with status ' + response.status));
      }

      const data = await response.json();
      renderPredictionResult(data);

      // Only once the analysis has succeeded is the Input Value series drawn,
      // using the exact values that were submitted.
      updateChart(FIELD_IDS.map(id => values[id]));
    } catch (err) {
      console.error('[FinInsight] Prediction request failed:', err);
      // Backend/network error: clear/hide the previous (or loading) prediction
      // state and chart series, and surface only the error message.
      resetPredictionSection('initial');
      const isNetworkError = err instanceof TypeError;
      showFormError(
        isNetworkError
          ? 'Unable to reach the prediction service. Please make sure the backend is running and try again.'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', () => {
      btnAnalyze.style.transform = 'scale(0.98)';
      setTimeout(() => { btnAnalyze.style.transform = ''; }, 150);
      handleAnalyze();
    });
  }

  // Expose global controller for future API wiring without DOM coupling
  window.FinInsight = {
    setTheme,
    updateChart,
    clearChart,
    getFormValues,
    setPredictionResult: renderPredictionResult
  };

  // Run initializations
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initForm();
    showFormError(null);
    resetPredictionSection('initial');
  });

})();