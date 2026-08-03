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

function doGet() {
  return jsonResponse({ok: true, service: "comp-climbing-style-tags", version: "2.0"});
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
    });
  } catch (error) {
    return jsonResponse({ok: false, message: String(error)});
  } finally {
    try { lock.releaseLock(); } catch (error) {}
  }
}
