"use strict";   // catches common JS mistakes early

// ── Global state ─────────────────────────────────────────────────
// allRows stores every prediction row so we can filter without re-fetching
let allRows = [];
let chartInstances = {};  // tracks Chart.js objects so we can destroy them on reset

// ── DOM references ────────────────────────────────────────────────
// Get HTML elements once at the top — faster than searching the DOM repeatedly
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const uploadSection  = document.getElementById("upload-section");
const loadingSection = document.getElementById("loading-section");
const resultsSection = document.getElementById("results-section");

// ── Drag and drop events ──────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();  // must prevent default to allow drop
  dropZone.style.background = "rgba(127,119,221,0.12)";
});
dropZone.addEventListener("dragleave", () => {
  dropZone.style.background = "";
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.background = "";
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── handleFile: the main entry point ─────────────────────────────
// Called whenever user picks or drops a CSV
function handleFile(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Please upload a .csv file");
    return;
  }

  // Switch screens: hide upload, show loading
  uploadSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");

  // Animate the loading step indicators
  animateLoadingSteps();

  // Build FormData — packages the file for HTTP upload
  const formData = new FormData();
  formData.append("file", file);  // must match request.files["file"] in Flask

  // Send to Flask and wait for JSON response
  fetch("/analyze", { method: "POST", body: formData })
    .then(res  => res.json())
    .then(data => {
      if (data.error) {
        alert("Error: " + data.error);
        resetApp();
        return;
      }
      renderResults(data);
    })
    .catch(err => {
      alert("Connection error: " + err.message);
      resetApp();
    });
}

// ── Loading step animation ────────────────────────────────────────
// Lights up each step label with a short delay so user sees progress
function animateLoadingSteps() {
  const ids    = ["ls1","ls2","ls3","ls4","ls5"];
  const delays = [0, 1000, 2500, 4500, 7000];  // ms between each step lighting up

  ids.forEach((id, i) => {
    setTimeout(() => {
      if (i > 0) document.getElementById(ids[i-1]).className = "lstep done";
      document.getElementById(id).className = "lstep active";
    }, delays[i]);
  });
}

// ── renderResults: master function ───────────────────────────────
// Receives the full JSON from Flask and calls each sub-renderer
function renderResults(data) {
  // Switch screens: hide loading, show results
  loadingSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  // Fill in the 6 stat cards
  document.getElementById("stat-acc").textContent     = data.accuracy + "%";
  document.getElementById("stat-auc").textContent     = data.auc ? data.auc + "%" : "N/A";
  document.getElementById("stat-rows").textContent    = data.total_rows.toLocaleString();
  document.getElementById("stat-feats").textContent   = data.feature_count;
  document.getElementById("stat-classes").textContent = data.class_count;
  document.getElementById("stat-smote").textContent   = data.smote_applied ? "✓ Applied" : "Skipped";

  // Render all 4 Chart.js charts
  renderMetricsChart(data.class_metrics);
  renderDistChart(data.dist_data);
  renderFeaturesChart(data.top_features);
  renderPredChart(data.dist_data);

  // Show SHAP plots (images from matplotlib via base64)
  if (data.shap_summary_img) {
    document.getElementById("shap-summary-img").src =
      "data:image/png;base64," + data.shap_summary_img;
  } else {
    document.getElementById("shap-summary-card").classList.add("hidden");
  }

  if (data.shap_waterfall_img) {
    document.getElementById("shap-waterfall-img").src =
      "data:image/png;base64," + data.shap_waterfall_img;
  } else {
    document.getElementById("shap-waterfall-card").classList.add("hidden");
  }

  // Build the predictions table
  allRows = data.rows;
  populateClassFilter(data.classes);
  filterTable();
}

// ── Chart helpers ─────────────────────────────────────────────────

// Destroy an existing chart before redrawing (prevents Chart.js "canvas in use" error)
function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

// Shared dark-theme options used by all charts
const darkGrid = { color: "#21262D" };
const darkTick = { color: "#8B949E", font: { size: 11 } };

// ── Chart 1: Precision / Recall / F1 grouped bar chart ───────────
function renderMetricsChart(metrics) {
  destroyChart("metrics");

  // Build a custom legend above the chart
  const legend = document.getElementById("legend-metrics");
  legend.innerHTML = [
    ["Precision","rgba(127,119,221,0.85)"],
    ["Recall",   "rgba(29,158,117,0.85)"],
    ["F1",       "rgba(239,159,39,0.85)"]
  ].map(([name, col]) =>
    `<span style="display:flex;align-items:center;gap:4px">
       <span style="width:10px;height:10px;border-radius:2px;background:${col};display:inline-block"></span>
       ${name}
     </span>`
  ).join("");

  chartInstances["metrics"] = new Chart(
    document.getElementById("chart-metrics"),
    {
      type: "bar",
      data: {
        labels: metrics.map(m => m.name),   // class names on x-axis
        datasets: [
          {
            label: "Precision",
            data: metrics.map(m => m.precision),
            backgroundColor: "rgba(127,119,221,0.85)",
          },
          {
            label: "Recall",
            data: metrics.map(m => m.recall),
            backgroundColor: "rgba(29,158,117,0.85)",
          },
          {
            label: "F1",
            data: metrics.map(m => m.f1),
            backgroundColor: "rgba(239,159,39,0.85)",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },  // we built a custom legend above
        scales: {
          x: { ticks: darkTick, grid: darkGrid },
          y: {
            min: 0, max: 100,
            ticks: { ...darkTick, callback: v => v + "%" },
            grid: darkGrid,
          },
        },
      },
    }
  );
}

// ── Chart 2: Traffic distribution doughnut ────────────────────────
function renderDistChart(dist) {
  destroyChart("dist");
  chartInstances["dist"] = new Chart(
    document.getElementById("chart-dist"),
    {
      type: "doughnut",
      data: {
        labels: dist.map(d => d.label),
        datasets: [{
          data: dist.map(d => d.count),
          backgroundColor: dist.map(d => d.color),
          borderWidth: 0,               // no white lines between slices
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "60%",                  // size of the hole in the middle
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#8B949E", font: { size: 11 }, padding: 12, boxWidth: 12 },
          },
          tooltip: {
            callbacks: {
              // custom tooltip: show count with comma formatting
              label: ctx => ctx.label + ": " + ctx.parsed.toLocaleString() + " flows"
            }
          }
        },
      },
    }
  );
}

// ── Chart 3: Feature importance horizontal bar ────────────────────
function renderFeaturesChart(features) {
  destroyChart("features");
  // sort ascending so most important is at the top
  const sorted = [...features].sort((a, b) => a.importance - b.importance);

  chartInstances["features"] = new Chart(
    document.getElementById("chart-features"),
    {
      type: "bar",
      data: {
        labels: sorted.map(f => f.name),
        datasets: [{
          label: "Importance",
          data: sorted.map(f => f.importance),
          // gradient opacity: top bars are most opaque
          backgroundColor: sorted.map((_, i) =>
            `rgba(127,119,221,${0.35 + 0.65 * (i / sorted.length)})`
          ),
          borderWidth: 0,
        }],
      },
      options: {
        indexAxis: "y",                 // makes it horizontal
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { ...darkTick, callback: v => v + "%" },
            grid: darkGrid,
          },
          y: {
            ticks: { color: "#8B949E", font: { size: 10, family: "monospace" } },
            grid: { display: false },
          },
        },
      },
    }
  );
}

// ── Chart 4: Predicted class distribution bar ─────────────────────
function renderPredChart(dist) {
  destroyChart("pred");
  const sorted = [...dist].sort((a, b) => b.count - a.count);

  chartInstances["pred"] = new Chart(
    document.getElementById("chart-pred"),
    {
      type: "bar",
      data: {
        labels: sorted.map(d => d.label),
        datasets: [{
          label: "Flows",
          data: sorted.map(d => d.count),
          backgroundColor: sorted.map(d => d.color + "CC"),  // CC = 80% opacity
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: darkTick, grid: darkGrid },
          y: { ticks: darkTick, grid: darkGrid },
        },
      },
    }
  );
}

// ── Flow predictions table ────────────────────────────────────────

function populateClassFilter(classes) {
  const sel = document.getElementById("filter-class");
  sel.innerHTML = '<option value="">All classes</option>';
  classes.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  });
}

function filterTable() {
  const cls     = document.getElementById("filter-class").value;
  const correct = document.getElementById("filter-correct").value;

  const filtered = allRows.filter(r => {
    if (cls && r.predicted !== cls) return false;
    if (correct === "correct" && !r.correct) return false;
    if (correct === "wrong"   &&  r.correct) return false;
    return true;
  });

  renderTable(filtered);
}

function renderTable(rows) {
  const tbody = document.getElementById("flow-tbody");
  tbody.innerHTML = "";   // clear existing rows

  rows.forEach(r => {
    const pill = `style="background:${r.color}22;color:${r.color};
                  border:1px solid ${r.color}44;
                  padding:2px 8px;border-radius:5px;font-size:11px"`;
    const pips = Array.from({length: 5}, (_, i) => {
      const lit = i < r.severity;
      const col = r.severity >= 4 ? "#E24B4A" : r.severity >= 2 ? "#EF9F27" : "#1D9E75";
      return `<span style="display:inline-block;width:5px;height:9px;border-radius:2px;
              margin-right:2px;background:${lit ? col : "#30363D"}"></span>`;
    }).join("");

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.row}</td>
      <td><span ${pill}>${r.actual}</span></td>
      <td><span ${pill}>${r.predicted}</span></td>
      <td>${r.confidence}%</td>
      <td>${pips}</td>
      <td class="${r.correct ? "match-ok" : "match-bad"}">${r.correct ? "✓" : "✗"}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("table-count").textContent =
    `Showing ${rows.length.toLocaleString()} of ${allRows.length.toLocaleString()} flows`;
}

// ── Download predictions as CSV ───────────────────────────────────
function downloadCSV() {
  const header = "Row,Actual,Predicted,Confidence,Severity,Correct\n";
  const body   = allRows.map(r =>
    `${r.row},"${r.actual}","${r.predicted}",${r.confidence},${r.severity},${r.correct}`
  ).join("\n");

  // Create a temporary download link and click it programmatically
  const blob = new Blob([header + body], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = "apt_predictions.csv";
  a.click();
  URL.revokeObjectURL(url);  // free memory after download
}

// ── Reset app back to upload screen ──────────────────────────────
function resetApp() {
  resultsSection.classList.add("hidden");
  loadingSection.classList.add("hidden");
  uploadSection.classList.remove("hidden");

  fileInput.value = "";   // clear file input so same file can be re-uploaded
  allRows = [];

  // Destroy all Chart.js instances to free memory
  Object.keys(chartInstances).forEach(k => {
    chartInstances[k].destroy();
    delete chartInstances[k];
  });

  // Reset loading step styles
  ["ls1","ls2","ls3","ls4","ls5"].forEach(id => {
    document.getElementById(id).className = "lstep";
  });

  // Reset SHAP images and un-hide their cards
  document.getElementById("shap-summary-img").src   = "";
  document.getElementById("shap-waterfall-img").src = "";
  document.getElementById("shap-summary-card").classList.remove("hidden");
  document.getElementById("shap-waterfall-card").classList.remove("hidden");
}