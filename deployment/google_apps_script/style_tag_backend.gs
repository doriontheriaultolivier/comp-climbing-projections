/**
 * Durable Google Drive + Sheet backend for the public boulder tagger.
 *
 * Required Script Property:
 *   STYLE_TAG_FOLDER_ID = destination Google Drive folder ID
 *
 * Deploy as a Web app, execute as the owner, and allow anyone to access it.
 * Put the resulting /exec URL in Streamlit secret STYLE_TAG_WEBHOOK_URL.
 */

function jsonResponse(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}

function doGet(event) {
  const action = event && event.parameter ? String(event.parameter.action || "") : "";
  if (action !== "list") {
    return jsonResponse({ok: true, service: "comp-climbing-style-tags", version: "3.0"});
  }
  try {
    const folder = getDestinationFolder();
    const spreadsheet = getOrCreateSheet(folder);
    const sheet = spreadsheet.getSheetByName("tags") || spreadsheet.getSheets()[0];
    const lastRow = sheet.getLastRow();
    if (lastRow < 2) return jsonResponse({ok: true, records: []});
    const requested = Number(event.parameter.limit || 1500);
    const limit = Math.max(1, Math.min(3000, requested));
    const firstRow = Math.max(2, lastRow - limit + 1);
    const values = sheet.getRange(firstRow, 1, lastRow - firstRow + 1, 11).getValues();
    const records = values.map(function(row) {
      let record = {};
      try { record = JSON.parse(row[10] || "{}"); } catch (error) {}
      record.image_file_id = row[8] || record.image_file_id || "";
      record.image_url = row[9] || record.image_url || "";
      record.image_public_url = row[8]
        ? "https://drive.google.com/uc?export=view&id=" + row[8]
        : (record.image_public_url || "");
      return record;
    });
    return jsonResponse({ok: true, records: records});
  } catch (error) {
    return jsonResponse({ok: false, message: String(error), records: []});
  }
}

function getDestinationFolder() {
  const folderId = PropertiesService.getScriptProperties().getProperty("STYLE_TAG_FOLDER_ID");
  if (!folderId) throw new Error("STYLE_TAG_FOLDER_ID is not configured");
  return DriveApp.getFolderById(folderId);
}

function getOrCreateSheet(folder) {
  const properties = PropertiesService.getScriptProperties();
  const storedId = properties.getProperty("STYLE_TAG_SHEET_ID");
  if (storedId) {
    try {
      return SpreadsheetApp.openById(storedId);
    } catch (error) {
      properties.deleteProperty("STYLE_TAG_SHEET_ID");
    }
  }
  const spreadsheet = SpreadsheetApp.create("Comp Climbing - Boulder Style Tags");
  const file = DriveApp.getFileById(spreadsheet.getId());
  file.moveTo(folder);
  properties.setProperty("STYLE_TAG_SHEET_ID", spreadsheet.getId());
  const sheet = spreadsheet.getSheets()[0];
  sheet.setName("tags");
  sheet.appendRow([
    "submitted_at_utc", "competition_date", "competition", "round",
    "gender_terrain", "boulder", "contributor", "confidence",
    "image_file_id", "image_url", "record_json"
  ]);
  sheet.setFrozenRows(1);
  return spreadsheet;
}

function cleanFileName(value) {
  return String(value || "boulder-image")
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 120) || "boulder-image";
}

function doPost(event) {
  const lock = LockService.getScriptLock();
  try {
    lock.waitLock(25000);
    const payload = JSON.parse(event.postData.contents || "{}");
    const record = payload.record || {};
    if (!record.competition || !record.boulder) {
      return jsonResponse({ok: false, message: "Competition and boulder are required"});
    }

    const folder = getDestinationFolder();
    let imageFile = null;
    if (payload.image_base64) {
      const bytes = Utilities.base64Decode(payload.image_base64);
      if (bytes.length > 10 * 1024 * 1024) {
        return jsonResponse({ok: false, message: "Image exceeds 10 MB"});
      }
      const mime = String(record.image_name || "").toLowerCase().endsWith(".png")
        ? "image/png" : "image/jpeg";
      const name = cleanFileName(
        [record.competition_date, record.competition, record.round,
         record.gender_terrain, record.boulder, record.image_name].filter(Boolean).join("_")
      );
      imageFile = folder.createFile(Utilities.newBlob(bytes, mime, name));
      imageFile.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);
    }

    const spreadsheet = getOrCreateSheet(folder);
    const sheet = spreadsheet.getSheetByName("tags") || spreadsheet.getSheets()[0];
    sheet.appendRow([
      record.submitted_at_utc || new Date().toISOString(),
      record.competition_date || "",
      record.competition || "",
      record.round || "",
      record.gender_terrain || "",
      record.boulder || "",
      record.contributor || "",
      record.confidence || "",
      imageFile ? imageFile.getId() : "",
      imageFile ? imageFile.getUrl() : "",
      JSON.stringify(record),
    ]);
    return jsonResponse({
      ok: true,
      message: "Saved to Google Drive",
      row: sheet.getLastRow(),
      image_file_id: imageFile ? imageFile.getId() : "",
      image_public_url: imageFile
        ? "https://drive.google.com/uc?export=view&id=" + imageFile.getId()
        : "",
    });
  } catch (error) {
    return jsonResponse({ok: false, message: String(error)});
  } finally {
    try { lock.releaseLock(); } catch (error) {}
  }
}
