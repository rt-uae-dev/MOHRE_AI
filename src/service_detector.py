import json
import os
from pathlib import Path

# Optional third-party imports; module works with standard library if unavailable
try:  # pragma: no cover - requests may not be installed in minimal environments
    import requests
except Exception:  # pragma: no cover
    requests = None

try:  # pragma: no cover - BeautifulSoup may not be installed
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

# Optional Gemini import
try:  # pragma: no cover - Gemini is optional
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

# Configure Gemini API if available
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key and genai:
    genai.configure(api_key=api_key)

# Location of a bundled list of MOHRE services used as an offline fallback
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_SERVICE_PATH = BASE_DIR / "data" / "mohre_services.json"


def _load_local_services() -> list[str]:
    """Load services from the bundled JSON file if available."""
    try:
        with LOCAL_SERVICE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [s for s in data if isinstance(s, str)]
    except Exception:  # pragma: no cover - failure falls back to default list
        pass
    return []


# Fallback list if fetching services fails entirely
DEFAULT_SERVICES = _load_local_services() or [
    "Work Permit Cancellation",
    "New Work Permit Application",
    "Contract Amendment",
    "Labour Card Issuance",
    "Salary Complaint",
    "Work Permit Renewal",
    "Temporary Work Permit",
    "Part-Time Work Permit",
    "Student Training Permit",
    "Juvenile Work Permit",
    "Employment Contract Registration",
    "Labour Card Cancellation",
    "Labour Card Renewal",
    "Bank Guarantee Refund",
    "Issue New Employment Visa",
    "Employment Visa Renewal",
    "Employment Visa Cancellation",
    "Labour Complaint",
    "Absconding Report",
    "Wage Protection System Registration",
    "Wage Protection System Amendment",
    "Wage Protection System Cancellation",
    "Establishment Card Renewal",
    "Establishment Registration",
    "Domestic Worker Permit",
    "Transfer Work Permit",
    "Work Injury Compensation",
    "End of Service Benefits Claim",
    "Quota Request",
    "Job Offer Registration",
    "Work Permit Replacement",
    "Probation Period Contract",
    "Mission Work Permit",
    "Family Residence Visa Sponsorship",
    "Occupational Health Card",
    "Certification of Loss of Passport",
    "Grievance Submission",
    "Housing Allowance Request",
    "Contract Termination",
    "Employment History Request",
    "Complaint Follow-up",
    "Salary Certificate Issuance",
    "Resignation Notice",
    "Return of Work Permit Deposit",
    "Establishment Data Update",
    "Emiratisation Certificate Request",
    "Part-time Job Approval",
    "Exemption from Bank Guarantee",
    "Work Place Inspection Request",
    "Annual Leave Approval",
]


def fetch_mohre_services() -> list[str]:
    """Fetch list of MOHRE services from the official website.

    The function attempts to scrape the live MOHRE services page. If the
    request fails (e.g. network unavailable, site blocked) or yields no
    results, a bundled offline list is returned instead.
    """
    services: list[str] = []
    url = "https://www.mohre.gov.ae/en/services.aspx"
    if requests and BeautifulSoup:
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a"):
                text = (link.get_text() or "").strip()
                if text and "services" not in text.lower() and len(text.split()) <= 8:
                    services.append(text)
        except Exception as e:  # pragma: no cover - network issues
            print(f"⚠️ Could not fetch MOHRE services: {e}")

    if not services:
        services = _load_local_services()

    return sorted(set(s for s in services if s))


def _simple_keyword_match(text: str, services: list) -> str:
    """Fallback fuzzy matching when Gemini is unavailable."""
    import re

    text_words = set(re.findall(r"\w+", text.lower()))
    best_service = "Unknown Service"
    best_score = 0.0

    for service in services:
        service_words = set(re.findall(r"\w+", service.lower()))
        if not service_words:
            continue
        score = len(service_words & text_words) / len(service_words)
        if score > best_score:
            best_score = score
            best_service = service

    return best_service if best_score > 0 else "Unknown Service"


def detect_service_from_email(email_text: str, services: list | None = None) -> str:
    """Determine which MOHRE service the email is requesting.

    Parameters
    ----------
    email_text: str
        The raw text of the email body.
    services: list | None
        Optional list of service names. If not provided, services will be
        fetched from the MOHRE website.

    Returns
    -------
    str
        The detected service name or "Unknown Service" if not determined.
    """
    services = services or fetch_mohre_services() or DEFAULT_SERVICES

    # Use Gemini if API key configured
    if api_key:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = (
                "You are an assistant helping to route customer emails to the correct MOHRE service.\n"
                f"Here is a list of available services: {services}\n"
                "Given the email below, reply with the single best matching service from the list.\n"
                f"Email: ```{email_text}```"
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini service detection failed: {e}")

    # Fallback to simple keyword match
    return _simple_keyword_match(email_text, services)
