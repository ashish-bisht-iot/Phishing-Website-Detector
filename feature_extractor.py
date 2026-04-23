"""
feature_extractor.py
Extracts 30 features from a URL for phishing detection.
"""

import re
import socket
from urllib.parse import urlparse

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

try:
    import tldextract
    TLD_AVAILABLE = True
except ImportError:
    TLD_AVAILABLE = False


SUSPICIOUS_TLDS = {
    'xyz', 'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'click',
    'info', 'biz', 'pw', 'cc', 'ru', 'cn', 'su', 'ws'
}

PHISHING_KEYWORDS = [
    'login', 'signin', 'sign-in', 'secure', 'verify', 'verification',
    'account', 'update', 'confirm', 'password', 'credential', 'banking',
    'paypal', 'ebay', 'amazon', 'apple', 'microsoft', 'google', 'netflix',
    'support', 'helpdesk', 'wallet', 'recover', 'suspend', 'alert'
]

TRUSTED_DOMAINS = {
    'google.com', 'amazon.com', 'microsoft.com', 'apple.com',
    'github.com', 'facebook.com', 'twitter.com', 'youtube.com',
    'wikipedia.org', 'linkedin.com', 'instagram.com', 'reddit.com',
    'netflix.com', 'paypal.com', 'ebay.com', 'stackoverflow.com'
}


def extract_features(url: str) -> dict:
    """
    Extract 30 features from a URL.
    Returns a dict with feature names and values.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    try:
        parsed = urlparse(url)
    except Exception:
        return _empty_features()

    hostname = parsed.hostname or ''
    path = parsed.path or ''
    query = parsed.query or ''
    full_url = url

    features = {}

    # ── URL-level features ──────────────────────────────────────────────────

    features['url_length'] = len(full_url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)

    features['has_https'] = 1 if parsed.scheme == 'https' else 0

    features['has_ip'] = 1 if _is_ip(hostname) else 0

    features['dot_count'] = full_url.count('.')
    features['hyphen_count'] = hostname.count('-')
    features['underscore_count'] = full_url.count('_')
    features['slash_count'] = full_url.count('/')
    features['question_mark_count'] = full_url.count('?')
    features['equal_count'] = full_url.count('=')
    features['at_symbol'] = 1 if '@' in full_url else 0
    features['double_slash'] = 1 if '//' in path else 0
    features['has_redirect'] = 1 if any(
        p in query.lower() for p in ['url=', 'goto=', 'redirect=', 'redir=']
    ) else 0

    features['digit_count_url'] = sum(c.isdigit() for c in full_url)
    features['digit_ratio'] = round(features['digit_count_url'] / max(len(full_url), 1), 4)

    features['special_char_count'] = sum(
        1 for c in full_url if c in '!#$%&\'*+/=?^`{|}~'
    )

    # ── Domain / TLD features ───────────────────────────────────────────────

    parts = hostname.replace('www.', '').split('.')
    tld = parts[-1].lower() if parts else ''
    features['tld'] = tld
    features['suspicious_tld'] = 1 if tld in SUSPICIOUS_TLDS else 0

    subdomain_parts = parts[:-2] if len(parts) > 2 else []
    features['subdomain_count'] = len(subdomain_parts)
    features['subdomain_length'] = len('.'.join(subdomain_parts))

    clean_host = hostname.replace('www.', '')
    features['is_trusted_domain'] = 1 if clean_host in TRUSTED_DOMAINS else 0

    # ── Keyword / content features ──────────────────────────────────────────

    url_lower = full_url.lower()
    kw_hits = [kw for kw in PHISHING_KEYWORDS if kw in url_lower]
    features['phishing_keyword_count'] = len(kw_hits)
    features['has_brand_keyword'] = 1 if any(
        b in url_lower for b in ['paypal', 'ebay', 'amazon', 'apple', 'microsoft', 'google', 'netflix']
    ) else 0

    # ── Structural features ─────────────────────────────────────────────────

    features['path_depth'] = path.count('/')
    features['has_port'] = 1 if parsed.port and parsed.port not in (80, 443) else 0
    features['has_fragment'] = 1 if parsed.fragment else 0
    features['has_query'] = 1 if query else 0

    # ── WHOIS / DNS (optional, adds latency) ────────────────────────────────

    features['domain_age_days'] = _get_domain_age(hostname) if WHOIS_AVAILABLE else -1
    features['dns_resolves'] = _check_dns(hostname)

    return features


def features_to_vector(features: dict) -> list:
    """
    Convert features dict to a numeric vector for the ML model.
    Order must match the training feature order.
    """
    NUMERIC_KEYS = [
        'url_length', 'hostname_length', 'path_length',
        'has_https', 'has_ip',
        'dot_count', 'hyphen_count', 'underscore_count',
        'slash_count', 'question_mark_count', 'equal_count',
        'at_symbol', 'double_slash', 'has_redirect',
        'digit_count_url', 'digit_ratio', 'special_char_count',
        'suspicious_tld', 'subdomain_count', 'subdomain_length',
        'is_trusted_domain',
        'phishing_keyword_count', 'has_brand_keyword',
        'path_depth', 'has_port', 'has_fragment', 'has_query',
        'domain_age_days', 'dns_resolves',
    ]
    return [features.get(k, 0) for k in NUMERIC_KEYS]


FEATURE_VECTOR_KEYS = [
    'url_length', 'hostname_length', 'path_length',
    'has_https', 'has_ip',
    'dot_count', 'hyphen_count', 'underscore_count',
    'slash_count', 'question_mark_count', 'equal_count',
    'at_symbol', 'double_slash', 'has_redirect',
    'digit_count_url', 'digit_ratio', 'special_char_count',
    'suspicious_tld', 'subdomain_count', 'subdomain_length',
    'is_trusted_domain',
    'phishing_keyword_count', 'has_brand_keyword',
    'path_depth', 'has_port', 'has_fragment', 'has_query',
    'domain_age_days', 'dns_resolves',
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_ip(hostname: str) -> bool:
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    return bool(re.match(pattern, hostname))


def _get_domain_age(hostname: str) -> int:
    """Returns domain age in days, or -1 on failure."""
    try:
        from datetime import datetime, timezone
        w = whois.whois(hostname)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation is None:
            return -1
        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - creation).days
        return max(age, 0)
    except Exception:
        return -1


def _check_dns(hostname: str) -> int:
    """Returns 1 if hostname resolves, 0 otherwise."""
    try:
        socket.gethostbyname(hostname)
        return 1
    except Exception:
        return 0


def _empty_features() -> dict:
    return {k: 0 for k in FEATURE_VECTOR_KEYS}


if __name__ == '__main__':
    test_urls = [
        'https://google.com',
        'http://paypa1-secure-login.xyz/account/verify?user=abc',
        'http://192.168.1.1/bank-login',
        'https://amazon.com/products',
        'http://update-apple-id.ml/signin/confirm?redirect=true',
    ]
    for u in test_urls:
        f = extract_features(u)
        v = features_to_vector(f)
        print(f"\n{u}")
        print(f"  ip={f['has_ip']}  https={f['has_https']}  tld={f['tld']}  "
              f"suspicious_tld={f['suspicious_tld']}  keywords={f['phishing_keyword_count']}  "
              f"subdomains={f['subdomain_count']}  trusted={f['is_trusted_domain']}")
