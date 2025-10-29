# gsheets.py
# Safe wrappers for Google Sheets read/update using gspread and a service account

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import logging

# Define the path to your credentials file
CREDENTIALS_FILE = "secrets/gsheets_creds.json"

# Define Google Sheets API scope
SCOPE = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive'
]

def get_gsheet_client():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPE)
    return gspread.authorize(creds)

def safe_read_gsheet(sheet_id, range_name):
    """
    Reads values from a Google Sheet and returns a DataFrame.
    range_name must be the worksheet title (tab name)
    """
    try:
        client = get_gsheet_client()
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.worksheet(range_name)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error reading Google Sheet: {e}")
        return pd.DataFrame()

def safe_update_gsheet(sheet_id, range_name, df):
    """
    Updates the given range in a Google Sheet with the contents of a DataFrame.
    range_name must be the worksheet title (tab name)
    """
    try:
        client = get_gsheet_client()
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.worksheet(range_name)
        worksheet.clear()
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        logging.info("Google Sheet updated successfully.")
    except Exception as e:
        logging.error(f"Error updating Google Sheet: {e}")
