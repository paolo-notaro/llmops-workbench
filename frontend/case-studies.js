function withoutTerminalPeriod(value) {
  return String(value).replace(/\.(?=\s*$)/, "");
}

function makeElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function appendInline(parent, value) {
  const marker = String.fromCharCode(96);
  withoutTerminalPeriod(value).split(marker).forEach((part, index) => {
    if (index % 2 === 1) {
      parent.appendChild(makeElement("code", "", part));
    } else {
      parent.appendChild(document.createTextNode(part));
    }
  });
}

function sectionId(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function renderIndex(studies) {
  const container = document.querySelector("#caseStudyList");
  container.replaceChildren();
  studies.forEach((study) => {
    const row = makeElement("a", "case-index-row");
    row.href = "/case-studies/" + study.slug;
    row.appendChild(makeElement("span", "case-number", study.number));
    const copy = makeElement("span", "case-index-copy");
    copy.appendChild(makeElement("strong", "", study.title));
    const summary = makeElement("small");
    appendInline(summary, study.summary);
    copy.appendChild(summary);
    row.appendChild(copy);
    const themes = makeElement("span", "case-themes");
    study.themes.forEach((theme) => themes.appendChild(makeElement("b", "", theme)));
    row.appendChild(themes);
    const arrow = makeElement("span", "case-arrow", "->");
    arrow.setAttribute("aria-hidden", "true");
    row.appendChild(arrow);
    container.appendChild(row);
  });
}

function appendBlocks(parent, blocks) {
  blocks.forEach((block) => {
    if (block.type === "list") {
      const list = makeElement("ul");
      block.items.forEach((item) => {
        const entry = makeElement("li");
        appendInline(entry, item);
        list.appendChild(entry);
      });
      parent.appendChild(list);
    } else {
      const paragraph = makeElement("p");
      appendInline(paragraph, block.text);
      parent.appendChild(paragraph);
    }
  });
}

function studyLink(study, direction) {
  if (!study) return makeElement("span");
  const link = makeElement("a", direction === "Next" ? "next" : "");
  link.href = "/case-studies/" + study.slug;
  link.appendChild(makeElement("span", "", direction));
  link.appendChild(makeElement("strong", "", study.title));
  return link;
}

function renderDetail(study, studies) {
  document.title = study.title + " | LLMOps Workbench";
  const currentIndex = studies.findIndex((item) => item.slug === study.slug);
  const container = document.querySelector("#caseStudyDetail");
  container.replaceChildren();
  const intro = makeElement("header", "case-detail-intro");
  const breadcrumb = makeElement("div");
  const allStudies = makeElement("a", "", "Case studies");
  allStudies.href = "/case-studies";
  breadcrumb.append(allStudies, makeElement("span", "", "/ " + study.number));
  intro.appendChild(breadcrumb);
  intro.appendChild(makeElement("p", "section-kicker", study.themes.join(" / ")));
  intro.appendChild(makeElement("h1", "", study.title));
  const summary = makeElement("p");
  appendInline(summary, study.summary);
  intro.appendChild(summary);
  container.appendChild(intro);

  const layout = makeElement("div", "case-detail-layout");
  const toc = makeElement("aside", "case-toc");
  toc.appendChild(makeElement("span", "", "Contents"));
  const tocNav = makeElement("nav");
  study.sections.forEach((section) => {
    const link = makeElement("a", "", section.heading);
    link.href = "#" + sectionId(section.heading);
    tocNav.appendChild(link);
  });
  toc.appendChild(tocNav);
  toc.appendChild(makeElement("small", "", "Sanitized abstraction / no private implementation details"));
  layout.appendChild(toc);

  const article = makeElement("article", "case-article");
  study.sections.forEach((section) => {
    const sectionNode = makeElement("section");
    sectionNode.id = sectionId(section.heading);
    sectionNode.appendChild(makeElement("h2", "", section.heading));
    appendBlocks(sectionNode, section.blocks);
    article.appendChild(sectionNode);
  });
  layout.appendChild(article);
  container.appendChild(layout);

  const pagination = makeElement("nav", "case-pagination");
  pagination.setAttribute("aria-label", "Case study pagination");
  pagination.appendChild(studyLink(currentIndex > 0 ? studies[currentIndex - 1] : null, "Previous"));
  pagination.appendChild(studyLink(currentIndex < studies.length - 1 ? studies[currentIndex + 1] : null, "Next"));
  container.appendChild(pagination);
}

async function initializeCaseStudies() {
  try {
    const response = await fetch("/assets/case-studies-data.json?v=1");
    if (!response.ok) throw new Error("HTTP " + response.status);
    const payload = await response.json();
    const list = document.querySelector("#caseStudyList");
    if (list) {
      renderIndex(payload.studies);
      return;
    }
    const slug = window.location.pathname.split("/").filter(Boolean)[1];
    const study = payload.studies.find((item) => item.slug === slug);
    const detail = document.querySelector("#caseStudyDetail");
    if (!study) {
      detail.replaceChildren();
      const missing = makeElement("div", "case-not-found");
      missing.appendChild(makeElement("h1", "", "Case study not found"));
      const back = makeElement("a", "", "Return to all studies");
      back.href = "/case-studies";
      missing.appendChild(back);
      detail.appendChild(missing);
      return;
    }
    renderDetail(study, payload.studies);
  } catch (error) {
    const target = document.querySelector("#caseStudyList") || document.querySelector("#caseStudyDetail");
    target.replaceChildren(makeElement("p", "inline-error", "Case studies could not be loaded"));
  }
}

initializeCaseStudies();
