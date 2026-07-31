const chatWindow = document.querySelector("#chatWindow");
const queryForm = document.querySelector("#queryForm");
const queryInput = document.querySelector("#queryInput");
const topK = document.querySelector("#topK");
const sendButton = document.querySelector("#sendButton");
const sourceList = document.querySelector("#sourceList");
const sourceCount = document.querySelector("#sourceCount");
const documentList = document.querySelector("#documentList");
const documentCount = document.querySelector("#documentCount");
const assistantPanel = document.querySelector(".assistant-panel");
const productSidebar = document.querySelector(".product-sidebar");
const desktopProductLayout = window.matchMedia("(min-width: 861px)");
const runtimeStatus = document.querySelector("#runtimeStatus");
const modeChips = [...document.querySelectorAll(".mode-chip")];
const documentTitles = new Map();

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));
}

function withoutTerminalPeriod(value) {
  return String(value).replace(/\.(?=\s*$)/, "");
}

function withAssistantTerminalPeriod(value) {
  const text = String(value).trim();
  return text && !text.endsWith(".") ? `${text}.` : text;
}

function readableCheckName(name) {
  return name.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function sourceAnchorId(docId) {
  return `source-${String(docId).replace(/[^a-z0-9_-]/gi, "-")}`;
}

function plainMarkdownText(value) {
  return String(value)
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^[-*+]\s+/gm, "")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

function renderCitations(value, documents = []) {
  const citationNumbers = new Map();
  documents.forEach((document, index) => {
    if (!citationNumbers.has(document.doc_id)) citationNumbers.set(document.doc_id, index + 1);
  });
  let fallbackNumber = documents.length + 1;
  return escapeHtml(value).replace(/\[doc:([a-z0-9_-]+)\]/g, (_, docId) => {
    const label = documentTitles.get(docId) || readableCheckName(docId);
    const number = citationNumbers.get(docId) || fallbackNumber++;
    return `<a class="citation-reference" href="#${sourceAnchorId(docId)}" aria-label="Citation ${number}: ${escapeHtml(label)}" title="${escapeHtml(label)}">[${number}]</a>`;
  });
}

function setRuntime(status, label, detail = "") {
  runtimeStatus.className = `runtime-status ${status}`;
  runtimeStatus.innerHTML = `<span class="status-dot ${status}"></span><span>${escapeHtml(label)}</span>${detail ? `<small>${escapeHtml(detail)}</small>` : ""}`;
}

function responseMetadata(payload) {
  const checks = Object.entries(payload.quality_checks || {});
  const passedChecks = checks.filter(([, passed]) => passed).length;
  return `<footer class="message-meta">
      <span>${escapeHtml(payload.provider)}</span><span>${Number(payload.latency_ms).toFixed(2)} ms</span>
      <span>${(payload.retrieved_docs || []).length} chunks</span>
      <span class="${passedChecks === checks.length ? "meta-pass" : "meta-review"}">${passedChecks}/${checks.length} live checks</span>
    </footer>
    <div class="quality-checks" aria-label="Live quality checks">
      ${checks.map(([name, passed]) => `<span class="${passed ? "pass" : "review"}"><i aria-hidden="true"></i>${escapeHtml(readableCheckName(name))}</span>`).join("")}
    </div>`;
}

function addMessage(role, text, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role} ${options.refusal ? "refusal" : ""}`.trim();
  const author = role === "user" ? "You" : "Assistant";
  const avatar = role === "user" ? "Y" : "A";
  const messageText = role === "assistant"
    ? renderCitations(withAssistantTerminalPeriod(text), options.payload?.retrieved_docs || [])
    : escapeHtml(text);
  article.innerHTML = `<div class="message-author"><span class="${role === "user" ? "user-avatar" : "assistant-avatar"}" aria-hidden="true">${avatar}</span><strong>${author}</strong></div>
    <p class="message-copy">${messageText}</p>${options.payload ? responseMetadata(options.payload) : ""}`;
  chatWindow.appendChild(article);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function loadDocuments() {
  try {
    const response = await fetch("/documents");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const documents = await response.json();
    documentCount.textContent = `${documents.length} files`;
    documents.forEach((document) => documentTitles.set(document.doc_id, document.title));
    documentList.innerHTML = documents.map((document) => `<article class="document-row">
        <span class="file-mark" aria-hidden="true">MD</span>
        <div><h3>${escapeHtml(document.title)}</h3><p>${escapeHtml(withoutTerminalPeriod(document.description))}</p>
        <small>${document.indexed ? "Indexed" : "Unavailable"} / ${document.chunk_count} ${document.chunk_count === 1 ? "chunk" : "chunks"}</small></div>
      </article>`).join("");
  } catch (error) {
    documentCount.textContent = "Unavailable";
    documentList.innerHTML = `<div class="inline-error"><strong>Document index unavailable</strong><p>Metadata could not be loaded / querying may still be available</p></div>`;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    setRuntime("ok", "Local runtime online", payload.provider);
  } catch (error) {
    setRuntime("bad", "Runtime unavailable");
  }
}

function renderSources(documents) {
  sourceCount.textContent = `${documents.length} ${documents.length === 1 ? "chunk" : "chunks"}`;
  if (!documents.length) {
    sourceList.innerHTML = `<div class="empty-state"><strong>No matching chunks</strong><p>The retriever returned no non-zero matches</p></div>`;
    return;
  }
  const seenDocumentIds = new Set();
  sourceList.innerHTML = documents.map((document, index) => {
    const score = Number(document.score);
    const width = Math.max(4, Math.round(score * 100));
    const title = documentTitles.get(document.doc_id) || document.title || document.doc_id;
    const sourceId = seenDocumentIds.has(document.doc_id) ? "" : ` id="${sourceAnchorId(document.doc_id)}"`;
    seenDocumentIds.add(document.doc_id);
    const excerpt = plainMarkdownText(document.text);
    return `<article class="source-row"${sourceId}>
        <header><div><small>Reference [${index + 1}]</small><h3>${escapeHtml(title)}</h3></div><strong>${score.toFixed(3)}</strong></header>
        <div class="score-track" aria-label="Similarity score ${score.toFixed(3)}"><span style="width: ${width}%"></span></div>
        <p>${escapeHtml(excerpt.slice(0, 240))}${excerpt.length > 240 ? "..." : ""}</p>
        <code>${escapeHtml(document.chunk_id)}</code>
      </article>`;
  }).join("");
}

async function submitQuery(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  addMessage("user", query);
  sendButton.disabled = true;
  sendButton.textContent = "Evaluating";
  try {
    const response = await fetch("/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: Number(topK.value) }),
    });
    if (!response.ok) throw new Error(`Request failed with HTTP ${response.status}`);
    const payload = await response.json();
    const isRefusal = payload.answer.toLowerCase().includes("cannot help");
    addMessage("assistant", payload.answer, { refusal: isRefusal, payload });
    renderSources(payload.retrieved_docs || []);
  } catch (error) {
    addMessage("assistant", `The request failed: ${String(error)}`, { refusal: true });
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Ask assistant";
  }
}

function syncAssistantHeight() {
  assistantPanel.style.height = desktopProductLayout.matches ? `${productSidebar.offsetHeight}px` : "";
}

if ("ResizeObserver" in window) {
  const sidebarResizeObserver = new ResizeObserver(syncAssistantHeight);
  sidebarResizeObserver.observe(productSidebar);
}
desktopProductLayout.addEventListener("change", syncAssistantHeight);
window.addEventListener("resize", syncAssistantHeight);

modeChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    modeChips.forEach((item) => item.classList.remove("active"));
    chip.classList.add("active");
    queryInput.value = chip.dataset.query;
    queryInput.focus();
  });
});

queryForm.addEventListener("submit", submitQuery);
loadDocuments();
checkHealth();
syncAssistantHeight();
