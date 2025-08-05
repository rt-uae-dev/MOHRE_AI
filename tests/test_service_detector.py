import sys
sys.path.append('src')

from service_detector import detect_service_from_email, fetch_mohre_services


def test_detect_service_keyword():
    email = "Hello, I would like to cancel my work permit as soon as possible."
    services = ["Work Permit Cancellation", "Issue New Work Permit"]
    result = detect_service_from_email(email, services=services)
    assert result == "Work Permit Cancellation"


def test_fetch_mohre_services_offline_fallback():
    services = fetch_mohre_services()
    assert "Work Permit Cancellation" in services
    assert len(services) > 10
