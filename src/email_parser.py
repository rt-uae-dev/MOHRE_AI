import imaplib
import email
import os
import time
import shutil
from email.header import decode_header
from email.utils import parseaddr
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_ADDRESS")
PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER")
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "data/raw/downloads")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def clean_filename(name):
    return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_")).rstrip()

def download_attachments(msg, subject_folder):
    attachments_saved = []
    for part in msg.walk():
        content_disposition = str(part.get("Content-Disposition"))
        if "attachment" in content_disposition:
            filename = part.get_filename()
            if filename:
                filename = clean_filename(filename)
                folder_path = os.path.join(DOWNLOAD_DIR, subject_folder)
                os.makedirs(folder_path, exist_ok=True)
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "wb") as f:
                    f.write(part.get_payload(decode=True))
                attachments_saved.append(filepath)
                print(f"📎 Downloaded attachment: {filepath}")
    return attachments_saved

def save_email_body(msg, subject_folder):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode(errors="ignore")
    else:
        body = msg.get_payload(decode=True).decode(errors="ignore")

    # Extract sender's email address
    from_field = msg.get("From", "")
    sender_email = parseaddr(from_field)[1]

    folder_path = os.path.join(DOWNLOAD_DIR, subject_folder)
    os.makedirs(folder_path, exist_ok=True)
    body_path = os.path.join(folder_path, "email_body.txt")
    with open(body_path, "w", encoding="utf-8") as f:
        if sender_email:
            f.write(f"Sender: {sender_email}\n")
        f.write(body)
    print(f"📝 Saved email body: {body_path}")
    return body_path

def fetch_and_store_emails(unseen_only=True, since_today=True):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    criteria = []
    if unseen_only:
        criteria.append("UNSEEN")
    if since_today:
        # Use a more reliable date format for 2025
        today = datetime.now().strftime("%d-%b-%Y")
        print(f"📅 Today's date: {today}")
        criteria.append(f'SINCE "{today}"')
    search_criteria = "(" + " ".join(criteria) + ")" if criteria else "ALL"

    print(f"📬 Searching emails with criteria: {search_criteria}")
    status, messages = mail.search(None, search_criteria)
    if status != "OK":
        print("❌ Failed to search emails.")
        mail.logout()
        return

    email_ids = messages[0].split()
    print(f"🔍 Found {len(email_ids)} email(s) matching criteria.")
    
    # If no emails found with current criteria, try without date restriction but still UNSEEN only
    if len(email_ids) == 0 and since_today and unseen_only:
        print("🔄 No unseen emails found with date restriction, trying without date filter but UNSEEN only...")
        search_criteria = "UNSEEN"
        print(f"📬 Searching emails with criteria: {search_criteria}")
        status, messages = mail.search(None, search_criteria)
        if status == "OK":
            email_ids = messages[0].split()
            print(f"🔍 Found {len(email_ids)} unseen email(s) without date restriction.")
    
    # If still no emails found, don't process any emails (don't fall back to ALL)
    if len(email_ids) == 0:
        print("✅ No unseen emails found. Skipping email processing.")
        mail.logout()
        return

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")
                subject_folder = clean_filename(subject) or "NoSubject"

                print(f"\n📥 Processing email: {subject}")
                save_email_body(msg, subject_folder)
                download_attachments(msg, subject_folder)
                mail.store(eid, '+FLAGS', '\\Seen')

    mail.logout()
    print("✅ Finished checking inbox.\n")

def cleanup_old_files(days_old=60):
    now = time.time()
    cutoff = now - (days_old * 86400)
    for folder in os.listdir(DOWNLOAD_DIR):
        folder_path = os.path.join(DOWNLOAD_DIR, folder)
        if os.path.isdir(folder_path):
            folder_time = os.path.getmtime(folder_path)
            if folder_time < cutoff:
                shutil.rmtree(folder_path)
                print(f"🗑️ Deleted old folder: {folder_path}")

# Optional: Continuous runner if run directly
if __name__ == "__main__":
    while True:
        print(f"\n🔄 Checking inbox at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        fetch_and_store_emails()
        cleanup_old_files(days_old=60)
        print("⏳ Sleeping for 5 minutes...\n")
        time.sleep(5 * 60)
