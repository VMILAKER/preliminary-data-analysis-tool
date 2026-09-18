import re

_DATE_FORMATS = [
    "%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S", "%Y%m%d", "%d.%m.%y",
    "%Y-%m-%d %H:%M", "%b %d %Y", "%B %d, %Y",
]

_PHONE_CLEAN_RE = re.compile(r"[\s\-\(\)\.]")

_PHONE_PATTERNS = [
    re.compile(r"^\+?7\d{10}$"), re.compile(r"^8\d{10}$"),
    re.compile(r"^\+?1\d{10}$"), re.compile(r"^\d{10,15}$"),
]

_EMAIL_RE = re.compile(r"^[\w\.\+\-]+@[\w\-]+(\.[\w\-]+)+$")

_URL_RE = re.compile(r"^https?://[^\s/$.?#][^\s]*$", re.IGNORECASE)
