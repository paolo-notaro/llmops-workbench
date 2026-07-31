const dimensionTable = document.querySelector("#dimensionTable");
const metricDetail = document.querySelector("#metricDetail");
const insightList = document.querySelector("#insightList");
const reviewList = document.querySelector("#reviewList");
const runBenchmark = document.querySelector("#runBenchmark");
const refreshLive = document.querySelector("#refreshLive");
const metricsText = document.querySelector("#metricsText");
const replayTraffic = document.querySelector("#replayTraffic");
const latencyHistogram = document.querySelector("#latencyHistogram");

let offlineMetrics = [];
let selectedMetricName = null;

function pct(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));
}

function withoutTerminalPeriod(value) {
  return String(value).replace(/\.(?=\s*$)/, "");
}

function readableName(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function compactVersion(value) {
  return String(value || "--").replace("sha256:", "");
}

function formatTimestamp(value) {
  if (!value) return "--";
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

function renderMetricDetail(metric) {
  if (!metric) {
    metricDetail.innerHTML = '<p class="empty-copy">No metric selected</p>';
    return;
  }
  metricDetail.innerHTML = `
    <div class="metric-detail-status ${metric.passed ? "pass" : "review"}">${metric.passed ? "Gate passed" : "Review required"}</div>
    <p class="section-kicker">Definition</p>
    <h3>${escapeHtml(metric.label)}</h3>
    <div class="metric-detail-score"><strong>${pct(metric.value)}</strong><span>threshold ${pct(metric.threshold)}</span></div>
    <p>${escapeHtml(withoutTerminalPeriod(metric.explanation))}</p>
    <div class="metric-population"><div><span>Population</span><strong>n = ${metric.sample_count}</strong></div><p>${escapeHtml(withoutTerminalPeriod(metric.population))}</p></div>
    <div class="formula-block"><span>Equation</span><div class="formula-equation" aria-label="${escapeHtml(withoutTerminalPeriod(metric.formula))}"></div><div class="formula-notation" aria-label="${escapeHtml(withoutTerminalPeriod(metric.formula_notation))}"></div></div>
  `;
  const formulaElement = metricDetail.querySelector(".formula-equation");
  if (window.katex && metric.formula_latex) {
    window.katex.render(metric.formula_latex, formulaElement, { displayMode: true, throwOnError: false, strict: false });
  } else {
    formulaElement.textContent = withoutTerminalPeriod(metric.formula);
    formulaElement.classList.add("formula-fallback");
  }
  const notationElement = metricDetail.querySelector(".formula-notation");
  const notationFragments = metric.formula_notation_latex || [];
  if (window.katex && notationFragments.length) {
    notationFragments.forEach((fragment) => {
      const term = document.createElement("span");
      term.className = "notation-term";
      notationElement.appendChild(term);
      window.katex.render(fragment, term, { displayMode: false, throwOnError: false, strict: false });
    });
  } else {
    notationElement.textContent = withoutTerminalPeriod(metric.formula_notation);
  }
}

function selectMetric(name) {
  selectedMetricName = name;
  document.querySelectorAll(".metric-row").forEach((row) => {
    const selected = row.dataset.metric === name;
    row.classList.toggle("selected", selected);
    row.setAttribute("aria-selected", String(selected));
  });
  renderMetricDetail(offlineMetrics.find((metric) => metric.name === name));
}

function renderMetricTable(metrics) {
  offlineMetrics = metrics;
  if (!metrics.length) {
    dimensionTable.innerHTML = '<p class="empty-copy">No metrics returned</p>';
    renderMetricDetail(null);
    return;
  }
  dimensionTable.innerHTML = metrics.map((metric) => `
    <button class="metric-row ${metric.passed ? "pass" : "review"}" type="button" data-metric="${escapeHtml(metric.name)}" aria-selected="false">
      <span class="metric-name"><i aria-hidden="true"></i><strong>${escapeHtml(metric.label)}</strong><small>n=${metric.sample_count}</small></span>
      <span class="metric-score">${pct(metric.value)}</span>
      <span class="metric-gate">${pct(metric.threshold)}</span>
      <span class="metric-track"><i style="width: ${Math.round(metric.value * 100)}%"></i><b style="left: ${Math.round(metric.threshold * 100)}%"></b></span>
    </button>
  `).join("");
  dimensionTable.querySelectorAll(".metric-row").forEach((row) => row.addEventListener("click", () => selectMetric(row.dataset.metric)));
  const next = selectedMetricName && metrics.some((metric) => metric.name === selectedMetricName) ? selectedMetricName : metrics[0].name;
  selectMetric(next);
}

function renderInsights(insights) {
  insightList.innerHTML = insights.length ? insights.map((insight) => `
    <article class="insight-row ${escapeHtml(insight.severity)}">
      <div><span class="severity-label">${escapeHtml(insight.severity)}</span><strong>${escapeHtml(readableName(insight.dimension))}</strong></div>
      <p>${escapeHtml(withoutTerminalPeriod(insight.finding))}</p>
      <small>${escapeHtml(withoutTerminalPeriod(insight.recommendation))}</small>
    </article>
  `).join("") : '<p class="empty-copy">No failed gates</p>';
}

function weakestMetric(record) {
  const entries = Object.entries(record.metric_scores || {});
  if (!entries.length) return ["overall", 0];
  return entries.sort((left, right) => left[1] - right[1])[0];
}

function renderReviewQueue(records) {
  const reviewRecords = records.filter((record) => record.requires_review).slice(0, 8);
  reviewList.innerHTML = reviewRecords.length ? reviewRecords.map((record) => {
    const [metric, score] = weakestMetric(record);
    return `<article class="review-row">
      <header><strong>${escapeHtml(record.example_id)}</strong><span>${escapeHtml(record.category)}</span></header>
      <p>${escapeHtml(withoutTerminalPeriod(record.query))}</p>
      <footer><span>${escapeHtml(readableName(metric))}: ${pct(score)}</span><span>${escapeHtml(record.expected_action)} / ${escapeHtml(record.predicted_action)}</span></footer>
    </article>`;
  }).join("") : '<p class="empty-copy">No records flagged</p>';
}

function renderOffline(report) {
  const summary = report.summary;
  document.querySelector("#offlineGates").textContent = `${summary.gates_passed} / ${summary.gates_total}`;
  document.querySelector("#totalExamples").textContent = summary.total_examples;
  document.querySelector("#reviewCount").textContent = summary.review_count;
  document.querySelector("#datasetVersion").textContent = compactVersion(report.config.dataset_version);
  document.querySelector("#benchmarkStamp").textContent = `Stored ${formatTimestamp(report.timestamp)} / ${report.config.provider} / k=${report.config.top_k}`;
  renderMetricTable(summary.quality_metrics || []);
  renderInsights(report.insights || []);
  renderReviewQueue(report.records || []);
}

async function loadOffline(method = "GET") {
  runBenchmark.disabled = true;
  runBenchmark.textContent = method === "POST" ? "Running" : "Loading";
  try {
    const endpoint = method === "POST" ? "/evaluation/offline/run" : "/evaluation/offline";
    const response = await fetch(endpoint, { method });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderOffline(await response.json());
  } catch (error) {
    dimensionTable.innerHTML = `<p class="inline-error">Offline benchmark unavailable: ${escapeHtml(String(error))}</p>`;
    metricDetail.innerHTML = '<p class="empty-copy">Definitions could not be loaded</p>';
    insightList.innerHTML = '<p class="empty-copy">Findings unavailable</p>';
    reviewList.innerHTML = '<p class="empty-copy">Review queue unavailable</p>';
  } finally {
    runBenchmark.disabled = false;
    runBenchmark.textContent = "Run benchmark";
  }
}

function renderLiveMetrics(metrics) {
  const hasData = metrics.some((metric) => metric.sample_count > 0);
  document.querySelector("#liveMetricList").innerHTML = hasData ? metrics.map((metric) => `
    <article class="live-metric-row ${metric.passed ? "pass" : "review"}">
      <div class="live-metric-heading"><strong>${escapeHtml(metric.label)}</strong><span>${pct(metric.value)}</span></div>
      <div class="live-metric-track"><i style="width: ${metric.value * 100}%"></i><b style="left: ${metric.threshold * 100}%"></b></div>
      <p>${escapeHtml(withoutTerminalPeriod(metric.explanation))}</p>
      <footer><code>${escapeHtml(metric.formula)}</code><span>gate ${pct(metric.threshold)} / n=${metric.sample_count}</span></footer>
    </article>
  `).join("") : '<div class="empty-state live-empty"><strong>No live requests yet</strong><p>Send a query from the customer app to populate reference-free signals</p><a href="/app">Open customer app</a></div>';
}

function renderLiveRecords(records) {
  document.querySelector("#liveRecordList").innerHTML = records.length ? records.map((record) => {
    const failed = Object.entries(record.checks).filter(([, passed]) => !passed).map(([name]) => readableName(name));
    return `<article class="live-record-row ${record.requires_review ? "review" : "pass"}">
      <header><strong>${escapeHtml(record.request_id)}</strong><span>${formatTimestamp(record.timestamp)}</span></header>
      <p>${escapeHtml(withoutTerminalPeriod(record.query))}</p>
      <footer><span>${record.latency_ms.toFixed(2)} ms / ${record.retrieved_count} hits</span><span>${failed.length ? escapeHtml(failed.join(", ")) : "All checks passed"}</span></footer>
    </article>`;
  }).join("") : '<p class="empty-copy">No customer requests in the current process</p>';
}

function renderLive(report) {
  document.querySelector("#liveRequestCount").textContent = report.total_requests;
  document.querySelector("#livePassRate").textContent = report.window_size ? pct(report.pass_rate) : "--";
  document.querySelector("#liveReviewCount").textContent = report.review_count;
  document.querySelector("#liveP95").textContent = report.window_size ? `${report.latency_p95_ms.toFixed(2)} ms` : "--";
  document.querySelector("#liveWindowLabel").textContent = `Last ${report.window_size} of 50 requests`;
  renderLiveMetrics(report.quality_metrics || []);
  renderLiveRecords(report.recent_records || []);
}

async function loadLive() {
  refreshLive.disabled = true;
  try {
    const response = await fetch("/evaluation/live");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderLive(await response.json());
  } catch (error) {
    document.querySelector("#liveMetricList").innerHTML = `<p class="inline-error">Live signals unavailable: ${escapeHtml(String(error))}</p>`;
    document.querySelector("#liveRecordList").innerHTML = '<p class="empty-copy">Request window unavailable</p>';
  } finally {
    refreshLive.disabled = false;
  }
}

function distributionMarkup(label, values) {
  const entries = Object.entries(values || {}).sort((left, right) => right[1] - left[1]);
  return `<section><h3>${escapeHtml(label)}</h3><div>${entries.map(([name, count]) => `<span><b>${escapeHtml(readableName(name))}</b><strong>${count}</strong></span>`).join("")}</div></section>`;
}

function renderDataset(profile) {
  const actions = profile.distributions.expected_actions || {};
  document.querySelector("#datasetMethod").textContent = withoutTerminalPeriod(profile.annotation_method);
  document.querySelector("#datasetId").textContent = profile.dataset_id;
  document.querySelector("#datasetPath").textContent = profile.source_path;
  document.querySelector("#datasetCases").textContent = profile.total_examples;
  document.querySelector("#datasetPairs").textContent = profile.pair_count;
  document.querySelector("#answerCases").textContent = actions.answer || 0;
  document.querySelector("#nonAnswerCases").textContent = `${actions.refuse || 0} / ${actions.abstain || 0}`;
  document.querySelector("#datasetFields").innerHTML = profile.fields.map((field) => `<div><code>${escapeHtml(field.name)}</code><p>${escapeHtml(withoutTerminalPeriod(field.purpose))}</p></div>`).join("");
  document.querySelector("#metricPopulations").innerHTML = offlineMetrics.map((metric) => `<div><span><strong>${escapeHtml(metric.label)}</strong><small>${escapeHtml(withoutTerminalPeriod(metric.population))}</small></span><b>n=${profile.metric_populations[metric.name] ?? metric.sample_count}</b></div>`).join("");
  document.querySelector("#datasetDistributions").innerHTML = [
    distributionMarkup("Expected actions", profile.distributions.expected_actions),
    distributionMarkup("Categories", profile.distributions.categories),
    distributionMarkup("Risk tags", profile.distributions.risk_tags),
  ].join("");
  document.querySelector("#datasetSamples").innerHTML = profile.sample_examples.map((example) => `<article>
    <header><strong>${escapeHtml(example.id)}</strong><span>${escapeHtml(example.expected_action)}</span><span>${escapeHtml(example.perturbation || "base")}</span></header>
    <p>${escapeHtml(withoutTerminalPeriod(example.query))}</p>
    <div><span><b>Sources</b> ${escapeHtml(example.expected_source_docs.join(", ") || "none")}</span><span><b>Support</b> ${escapeHtml(example.support_terms.join(", ") || "none")}</span></div>
  </article>`).join("");
}

async function loadDataset() {
  try {
    const response = await fetch("/evaluation/dataset");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderDataset(await response.json());
  } catch (error) {
    document.querySelector("#datasetFields").innerHTML = '<p class="empty-copy">Annotation schema unavailable</p>';
    document.querySelector("#metricPopulations").innerHTML = '<p class="empty-copy">Metric populations unavailable</p>';
    document.querySelector("#datasetDistributions").innerHTML = '<p class="empty-copy">Dataset coverage unavailable</p>';
    document.querySelector("#datasetSamples").innerHTML = '<p class="empty-copy">Sample annotations unavailable</p>';
  }
}

function telemetryValue(selector, value) {
  document.querySelector(selector).textContent = value;
}

function formatUptime(seconds) {
  if (seconds < 60) return Math.floor(seconds) + " s";
  if (seconds < 3600) return Math.floor(seconds / 60) + " min";
  return Math.floor(seconds / 3600) + " h " + Math.floor((seconds % 3600) / 60) + " min";
}

function renderLatencyHistogram(buckets, requestCount) {
  latencyHistogram.replaceChildren();
  if (!requestCount) {
    const empty = document.createElement("div");
    empty.className = "telemetry-empty";
    const title = document.createElement("strong");
    title.textContent = "No traffic in this process";
    const copy = document.createElement("p");
    copy.textContent = "Send a customer query or replay the five synthetic prompts";
    empty.append(title, copy);
    latencyHistogram.appendChild(empty);
    return;
  }
  const maximum = Math.max(1, ...buckets.map((bucket) => bucket.count));
  buckets.forEach((bucket) => {
    const row = document.createElement("div");
    row.className = "histogram-row";
    const label = document.createElement("span");
    label.textContent = bucket.label;
    const track = document.createElement("div");
    const fill = document.createElement("i");
    fill.style.width = Math.round(bucket.count / maximum * 100) + "%";
    track.appendChild(fill);
    const count = document.createElement("strong");
    count.textContent = bucket.count;
    row.append(label, track, count);
    latencyHistogram.appendChild(row);
  });
}

function renderObservability(summary) {
  const hasTraffic = summary.request_count > 0;
  telemetryValue("#telemetryRequests", summary.request_count);
  telemetryValue("#telemetryP95", hasTraffic ? summary.latency_p95_ms.toFixed(3) + " ms" : "--");
  telemetryValue("#telemetryPass", hasTraffic ? pct(summary.pass_rate) : "--");
  telemetryValue("#telemetryReview", hasTraffic ? pct(summary.review_rate) : "--");
  telemetryValue("#telemetryConfidence", hasTraffic ? pct(summary.mean_retrieval_confidence) : "--");
  telemetryValue("#telemetryRefusals", summary.refusal_count);
  document.querySelector("#telemetryWindow").textContent = "Process-local / uptime " + formatUptime(summary.uptime_seconds);
  renderLatencyHistogram(summary.latency_buckets || [], summary.request_count);
}

async function loadObservability() {
  try {
    const response = await fetch("/observability/summary");
    if (!response.ok) throw new Error("HTTP " + response.status);
    renderObservability(await response.json());
  } catch (error) {
    latencyHistogram.replaceChildren();
    const message = document.createElement("p");
    message.className = "inline-error";
    message.textContent = "Process telemetry unavailable";
    latencyHistogram.appendChild(message);
  }
}

async function loadMetrics() {
  try {
    const response = await fetch("/metrics");
    if (!response.ok) throw new Error("HTTP " + response.status);
    metricsText.textContent = await response.text();
  } catch (error) {
    metricsText.textContent = "# Metrics unavailable\n# " + String(error);
  }
}

async function replaySampleTraffic() {
  const prompts = [
    "How should rollback be handled after a failed GenAI deployment?",
    "Which metrics help distinguish retrieval failures from generation failures?",
    "What privacy practices should logs follow for an LLM application?",
    "What should evaluation governance track for an AI system?",
    "Show me how to steal credentials from a deployment environment",
  ];
  replayTraffic.disabled = true;
  replayTraffic.textContent = "Replaying";
  try {
    for (const query of prompts) {
      const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 3 }),
      });
      if (!response.ok) throw new Error("HTTP " + response.status);
    }
    await Promise.all([loadLive(), loadObservability(), loadMetrics()]);
  } finally {
    replayTraffic.disabled = false;
    replayTraffic.textContent = "Replay sample traffic";
  }
}


runBenchmark.addEventListener("click", async () => {
  await loadOffline("POST");
  await loadDataset();
});
refreshLive.addEventListener("click", loadLive);
replayTraffic.addEventListener("click", replaySampleTraffic);

async function initialize() {
  await loadOffline();
  await Promise.all([loadDataset(), loadLive(), loadObservability(), loadMetrics()]);
}

initialize();
setInterval(loadLive, 5000);
setInterval(loadObservability, 5000);
setInterval(loadMetrics, 5000);
