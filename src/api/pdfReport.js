import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import { plantData } from "../data/plants";

// ─── Color palette ─────────────────────────────────────────────────────────
const GREEN_DARK  = [30, 77, 36];
const GREEN_MID   = [45, 106, 53];
const GREEN_LIGHT = [211, 235, 214];
const GREEN_PALE  = [237, 247, 238];
const TEXT_DARK   = [26, 58, 31];
const TEXT_MID    = [61, 92, 66];
const TEXT_MUTED  = [107, 136, 114];
const WHITE       = [255, 255, 255];
const GOLD        = [180, 140, 20];
const AMBER_BG    = [255, 248, 225];
const AMBER_BORDER= [220, 170, 0];
const AMBER_TEXT  = [120, 85, 0];

// ─── Helpers ───────────────────────────────────────────────────────────────
function ratingLabel(pct) {
  if (pct >= 85) return "Excellent";
  if (pct >= 70) return "Good";
  if (pct >= 55) return "Moderate";
  return "Low";
}

function ratingStars(pct) {
  if (pct >= 85) return "[5/5]";
  if (pct >= 70) return "[4/5]";
  if (pct >= 55) return "[3/5]";
  return "[2/5]";
}

function ratingColor(pct) {
  if (pct >= 85) return [39, 174, 96];
  if (pct >= 70) return [52, 152, 219];
  if (pct >= 55) return [230, 126, 34];
  return [192, 57, 43];
}

function sectionHeader(doc, text, y, PW, M) {
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...GREEN_DARK);
  doc.text(text, M, y);
  doc.setDrawColor(...GREEN_MID);
  doc.setLineWidth(0.5);
  doc.line(M, y + 3, PW - M, y + 3);
  return y + 10;
}

// ─── Main export ───────────────────────────────────────────────────────────
export async function generatePDFReport(results) {
  const { ranked, selectedTree, city, climate, soilData } = results;
  const selectedResult = ranked.find((r) => r.plant === selectedTree);
  const bestResult     = ranked[0];
  const info           = plantData[selectedTree];

  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const PW  = doc.internal.pageSize.getWidth();   // 210
  const PH  = doc.internal.pageSize.getHeight();  // 297
  const M   = 18;

  // ══════════════════════════════════════════════
  //  PAGE 1
  // ══════════════════════════════════════════════

  // ── Header band ────────────────────────────────
  doc.setFillColor(...GREEN_DARK);
  doc.rect(0, 0, PW, 42, "F");

  // Logo circle (plain — no emoji)
  doc.setFillColor(...GREEN_MID);
  doc.circle(M + 6, 21, 7, "F");
  doc.setFontSize(8);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...WHITE);
  doc.text("GP", M + 6, 23.5, { align: "center" });

  // App name
  doc.setFontSize(20);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...WHITE);
  doc.text("GreenPredict AI", M + 18, 18);

  // Subtitle
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(180, 220, 185);
  doc.text("Tree Survival Prediction & Recommendation Report", M + 18, 26);

  // Date (top-right)
  const dateStr = new Date().toLocaleDateString("en-IN", {
    day: "2-digit", month: "short", year: "numeric",
  });
  doc.setFontSize(8);
  doc.setTextColor(180, 220, 185);
  doc.text("Generated: " + dateStr, PW - M, 26, { align: "right" });

  // ── Selected tree card ─────────────────────────
  let y = 54;

  doc.setFillColor(...GREEN_PALE);
  doc.roundedRect(M, y - 6, PW - 2 * M, 54, 4, 4, "F");
  doc.setDrawColor(...GREEN_LIGHT);
  doc.setLineWidth(0.4);
  doc.roundedRect(M, y - 6, PW - 2 * M, 54, 4, 4, "S");

  // Tree name
  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...GREEN_DARK);
  doc.text(selectedTree, M + 6, y + 5);

  // Scientific name
  doc.setFontSize(9);
  doc.setFont("helvetica", "italic");
  doc.setTextColor(...TEXT_MID);
  doc.text(info.scientificName, M + 6, y + 13);

  // Description (left column)
  doc.setFontSize(8.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(...TEXT_MID);
  const descLines = doc.splitTextToSize(info.description, PW - 2 * M - 72);
  doc.text(descLines, M + 6, y + 22);

  // Survival % box (right side)
  const boxX = PW - M - 60;
  doc.setFillColor(...GREEN_MID);
  doc.roundedRect(boxX, y - 2, 54, 46, 4, 4, "F");

  doc.setFontSize(7.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(...WHITE);
  doc.text("Predicted Survival Rate", boxX + 27, y + 8, { align: "center" });

  doc.setFontSize(32);
  doc.setFont("helvetica", "bold");
  doc.text(selectedResult.probability.toFixed(0) + "%", boxX + 27, y + 26, { align: "center" });

  doc.setFontSize(8);
  doc.setFont("helvetica", "normal");
  doc.text(ratingLabel(selectedResult.probability) + " " + ratingStars(selectedResult.probability),
    boxX + 27, y + 35, { align: "center" });

  // Rank badge
  doc.setFillColor(...GOLD);
  doc.circle(boxX + 27, y + 41, 5, "F");
  doc.setFontSize(7);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(255, 255, 255);
  doc.text("#" + selectedResult.rank, boxX + 27, y + 43, { align: "center" });

  y += 60;

  // ── Better recommendation banner ───────────────
  if (bestResult.plant !== selectedTree) {
    doc.setFillColor(...AMBER_BG);
    doc.roundedRect(M, y, PW - 2 * M, 14, 3, 3, "F");
    doc.setDrawColor(...AMBER_BORDER);
    doc.setLineWidth(0.3);
    doc.roundedRect(M, y, PW - 2 * M, 14, 3, 3, "S");

    doc.setFontSize(8.5);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(...AMBER_TEXT);
    doc.text(
      "Better recommendation: " + bestResult.plant +
      " (" + bestResult.probability.toFixed(0) + "% survival) — ranked #1 for your conditions",
      M + 5, y + 9
    );
    y += 20;
  }

  // ── Soil & Environmental Data ──────────────────
  y += 4;
  y = sectionHeader(doc, "Soil & Environmental Data", y, PW, M);

  autoTable(doc, {
    startY: y,
    margin: { left: M, right: M },
    head: [],
    body: [
      ["City",          city,                               "Rainfall",        climate.rainfall + " mm/year"],
      ["Temperature",   climate.avg_temp + " deg C",        "Soil Type",       soilData.soil_type],
      ["pH Value",      soilData.pH.toFixed(2),             "Organic Carbon",  soilData.organic_carbon + "%"],
      ["Nitrogen (N)",  soilData.nitrogen + " kg/ha",       "Phosphorus (P)",  soilData.phosphorus + " kg/ha"],
      ["Potassium (K)", soilData.potassium + " kg/ha",      "EC",              soilData.ec + " dS/m"],
    ],
    columnStyles: {
      0: { fontStyle: "bold", textColor: TEXT_MID,  cellWidth: 38, fillColor: GREEN_PALE },
      1: { textColor: TEXT_DARK, cellWidth: 48 },
      2: { fontStyle: "bold", textColor: TEXT_MID,  cellWidth: 38, fillColor: GREEN_PALE },
      3: { textColor: TEXT_DARK, cellWidth: 48 },
    },
    styles: {
      fontSize: 9,
      cellPadding: { top: 4, bottom: 4, left: 5, right: 5 },
      lineColor: GREEN_LIGHT,
      lineWidth: 0.3,
      font: "helvetica",
    },
    alternateRowStyles: { fillColor: WHITE },
    theme: "grid",
  });

  y = doc.lastAutoTable.finalY + 12;

  // ── About the Plant ────────────────────────────
  y = sectionHeader(doc, "About " + selectedTree, y, PW, M);

  autoTable(doc, {
    startY: y,
    margin: { left: M, right: M },
    head: [],
    body: [
      ["Soil Requirements", info.soil],
      ["Climate",           info.climate],
      ["Key Benefits",      info.benefits.join(" | ")],
    ],
    columnStyles: {
      0: { fontStyle: "bold", textColor: TEXT_MID, cellWidth: 42, fillColor: GREEN_PALE },
      1: { textColor: TEXT_DARK },
    },
    styles: {
      fontSize: 9,
      cellPadding: { top: 5, bottom: 5, left: 5, right: 5 },
      lineColor: GREEN_LIGHT,
      lineWidth: 0.3,
      font: "helvetica",
    },
    theme: "grid",
  });

  // ══════════════════════════════════════════════
  //  PAGE 2
  // ══════════════════════════════════════════════
  doc.addPage();

  // Header stripe
  doc.setFillColor(...GREEN_DARK);
  doc.rect(0, 0, PW, 20, "F");
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...WHITE);
  doc.text("All Tree Recommendations — Ranked by Survival Rate", M, 13);

  y = 30;
  doc.setFontSize(8.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(...TEXT_MUTED);
  doc.text(
    "Results for " + city + "  |  " + soilData.soil_type + " soil  |  pH " + soilData.pH,
    M, y
  );
  y += 8;

  // Rankings table — NO emojis, no special chars
  const tableBody = ranked.map((item) => {
    const p = plantData[item.plant];
    const isChosen = item.plant === selectedTree;
    return [
      "#" + item.rank,
      item.plant + "\n" + p.scientificName,
      item.probability.toFixed(1) + "%",
      ratingLabel(item.probability) + " " + ratingStars(item.probability),
      p.benefits.slice(0, 2).join(", "),
      isChosen ? "Your Choice" : "",
    ];
  });

  autoTable(doc, {
    startY: y,
    margin: { left: M, right: M },
    head: [["Rank", "Tree", "Survival", "Rating", "Key Benefits", ""]],
    body: tableBody,
    columnStyles: {
      0: { cellWidth: 12, halign: "center", fontStyle: "bold" },
      1: { cellWidth: 44 },
      2: { cellWidth: 20, halign: "center", fontStyle: "bold" },
      3: { cellWidth: 30 },
      4: { cellWidth: 56 },
      5: { cellWidth: 22, halign: "center", fontStyle: "italic" },
    },
    headStyles: {
      fillColor: GREEN_MID,
      textColor: WHITE,
      fontStyle: "bold",
      fontSize: 9,
      cellPadding: { top: 5, bottom: 5, left: 4, right: 4 },
      font: "helvetica",
    },
    bodyStyles: {
      fontSize: 8.5,
      cellPadding: { top: 5, bottom: 5, left: 4, right: 4 },
      lineColor: GREEN_LIGHT,
      lineWidth: 0.3,
      textColor: TEXT_DARK,
      font: "helvetica",
    },
    alternateRowStyles: { fillColor: GREEN_PALE },
    didParseCell(data) {
      if (data.section !== "body") return;
      const raw = data.row.raw;

      // Highlight chosen row
      if (raw && raw[5] === "Your Choice") {
        data.cell.styles.fillColor = [215, 245, 218];
      }
      // Gold rank #1
      if (data.column.index === 0 && data.row.index === 0) {
        data.cell.styles.textColor = GOLD;
        data.cell.styles.fontSize  = 12;
      }
      // Colour-coded survival %
      if (data.column.index === 2) {
        const pct = parseFloat(data.cell.text[0]);
        data.cell.styles.textColor = ratingColor(pct);
      }
      // Italic green "Your Choice"
      if (data.column.index === 5 && raw[5] === "Your Choice") {
        data.cell.styles.textColor = GREEN_MID;
      }
    },
    theme: "grid",
  });

  y = doc.lastAutoTable.finalY + 14;

  // ── Methodology box ────────────────────────────
  const boxH = 40;
  doc.setFillColor(...GREEN_PALE);
  doc.roundedRect(M, y, PW - 2 * M, boxH, 4, 4, "F");
  doc.setDrawColor(...GREEN_LIGHT);
  doc.setLineWidth(0.3);
  doc.roundedRect(M, y, PW - 2 * M, boxH, 4, 4, "S");

  // Left accent bar
  doc.setFillColor(...GREEN_MID);
  doc.roundedRect(M, y, 3, boxH, 2, 2, "F");

  doc.setFontSize(10);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...GREEN_DARK);
  doc.text("How We Calculate Survival Rates", M + 8, y + 10);

  doc.setFontSize(8);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(...TEXT_MID);
  const calcText =
    "Our AI model (Random Forest Classifier, 300 decision trees) analyzes soil composition " +
    "(type & pH), nutrient content (Nitrogen, Phosphorus, Potassium), organic carbon, " +
    "electrical conductivity (EC), and city-specific climate data (temperature & rainfall). " +
    "Survival rates: 85%+ = Excellent, 70-84% = Good, 55-69% = Moderate, below 55% = Low. " +
    "Model trained on 3,200 botanically-accurate synthetic samples across 8 plant species.";
  const calcLines = doc.splitTextToSize(calcText, PW - 2 * M - 14);
  doc.text(calcLines, M + 8, y + 18);

  // ── Footer — both pages ────────────────────────
  const totalPages = doc.internal.getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p);
    doc.setFillColor(...GREEN_DARK);
    doc.rect(0, PH - 12, PW, 12, "F");
    doc.setFontSize(7.5);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...WHITE);
    doc.text(
      "GreenPredict AI  |  Plant Survival Prediction System  |  Maharashtra, India",
      M, PH - 5
    );
    doc.text("Page " + p + " of " + totalPages, PW - M, PH - 5, { align: "right" });
  }

  // ── Save ───────────────────────────────────────
  const filename = "GreenPredict_" + selectedTree + "_" + city + "_" + Date.now() + ".pdf";
  doc.save(filename);
  return filename;
}