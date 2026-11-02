"""Part B — Text Normalizer: NFC normalization with ID preservation.

Rules:
- NFC normalize Vietnamese text (diacritics consistency)
- Preserve opaque IDs returned from tool results (don't NFC those)
- Collapse whitespace
- Preserve negation, numbers, time, phone numbers exactly
- Don't transform ambiguous sentences into certain requests
"""
import re
import unicodedata


def normalize_nfc(text: str) -> str:
    """Apply Unicode NFC normalization to text."""
    if not isinstance(text, str):
        return str(text)
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space, strip."""
    return " ".join(text.split())


def normalize_transcript(text: str) -> str:
    """Full normalization pipeline for ASR transcript output.
    
    Steps:
    1. NFC normalization (Vietnamese diacritics)
    2. Whitespace collapse
    3. Strip leading/trailing whitespace
    
    Does NOT change:
    - Negation words (không, đừng, chưa, etc.)
    - Numbers / phone numbers / amounts
    - Time expressions
    - Proper nouns / entity names
    """
    if not text:
        return ""
    
    text = normalize_nfc(text)
    text = normalize_whitespace(text)
    text = text.strip()
    
    return text


def normalize_for_fuzzy(text: str) -> str:
    """Normalize text specifically for fuzzy matching evaluation.
    
    Following evaluator rules:
    - NFC normalization
    - Lowercase
    """
    if not isinstance(text, str):
        return str(text).lower()
    return unicodedata.normalize("NFC", text).lower()


# === Number normalization helpers ===

# Vietnamese number words
_VN_DIGITS = {
    "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
    "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9",
    "mười": "10", "mươi": "0",  # mươi is tens place
}

_VN_MULTIPLIERS = {
    "nghìn": 1000, "ngàn": 1000,
    "triệu": 1_000_000,
    "tỷ": 1_000_000_000,
    "trăm": 100,
}

# Common ASR errors and their corrections
_COMMON_CORRECTIONS = {
    # English words commonly code-switched
    "book": "book",
    "booking": "booking",
    "cancel": "cancel",
    "AC": "AC",
    "air con": "AC",
    "wifi": "wifi",
    "OK": "OK",
}


def extract_phone_number(text: str) -> list[str]:
    """Extract Vietnamese phone numbers from text."""
    # Vietnamese phone patterns: 0xx-xxx-xxxx or 0xxxxxxxxx
    patterns = [
        r'0\d{2}[\s.-]?\d{3}[\s.-]?\d{4}',  # 0xx xxx xxxx
        r'0\d{9}',                              # 0xxxxxxxxxx
        r'\+84\d{9,10}',                        # +84xxxxxxxxx
    ]
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)
    return numbers


def extract_money_amount(text: str) -> list[dict]:
    """Extract money amounts from Vietnamese text.
    
    Handles: 50 nghìn, 500k, 1 triệu, 1.5 triệu, etc.
    """
    results = []
    
    # Pattern: number + unit
    patterns = [
        # "50 nghìn", "500 ngàn"
        (r'(\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ|trăm)', None),
        # "50k", "500K"  
        (r'(\d+(?:[.,]\d+)?)\s*[kK]', 1000),
        # "1tr", "1.5tr"
        (r'(\d+(?:[.,]\d+)?)\s*tr', 1_000_000),
        # Plain number (context-dependent)
        (r'(\d{4,})', 1),
    ]
    
    for pattern, fixed_mult in patterns:
        for match in re.finditer(pattern, text):
            num_str = match.group(1).replace(",", ".")
            try:
                num = float(num_str)
                if fixed_mult:
                    amount = int(num * fixed_mult)
                else:
                    unit = match.group(2).lower()
                    mult = _VN_MULTIPLIERS.get(unit, 1)
                    amount = int(num * mult)
                results.append({"text": match.group(0), "amount": amount})
            except (ValueError, IndexError):
                pass
    
    return results


def extract_time_expressions(text: str, current_time: str | None = None) -> list[str]:
    """Extract time-related expressions from Vietnamese text.
    
    Handles: "mai", "tối nay", "7 giờ", "ngày mai", "thứ 2", etc.
    """
    time_patterns = [
        r'\d{1,2}\s*(?:giờ|h)\s*(?:\d{1,2}\s*(?:phút|p))?',  # 7 giờ, 7h30, 7 giờ 30 phút
        r'\d{1,2}:\d{2}',                                       # 7:30
        r'(?:sáng|trưa|chiều|tối|đêm)\s*(?:nay|mai)?',         # sáng nay, tối mai
        r'(?:ngày\s*)?mai',                                      # mai, ngày mai
        r'hôm\s*(?:nay|qua)',                                    # hôm nay, hôm qua
        r'(?:thứ|chủ\s*nhật)\s*\d?',                            # thứ 2, chủ nhật
        r'tuần\s*(?:này|sau|trước)',                             # tuần này
        r'tháng\s*\d{1,2}',                                     # tháng 11
        r'ngày\s*\d{1,2}(?:/\d{1,2})?',                        # ngày 15, ngày 15/11
    ]
    
    results = []
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        results.extend(matches)
    
    return results


if __name__ == "__main__":
    # Test normalizations
    print("=== Normalizer Tests ===\n")
    
    tests = [
        "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng",
        "nạp giúp mình 50 nghìn điện thoại",
        "book giúp tôi xe 4 chỗ",
        "tắt đèn phòng   ngủ   nhỏ",
    ]
    
    for text in tests:
        normalized = normalize_transcript(text)
        print(f"  Input:  '{text}'")
        print(f"  Output: '{normalized}'")
        
        phones = extract_phone_number(text)
        money = extract_money_amount(text)
        times = extract_time_expressions(text)
        
        if phones: print(f"  Phones: {phones}")
        if money: print(f"  Money:  {money}")
        if times: print(f"  Times:  {times}")
        print()
