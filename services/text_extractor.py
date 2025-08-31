import re

class TextExtractor:
    def find_phone(self, text):
        """Extract phone number from text."""
        patterns = [
            r"(?:SĐT|Tel|Phone|Hotline)[\s:]*(\d{9,11})",
            r"((?:\+84|0)(?:3[2-9]|5[6|8|9]|7[0|6-9]|8[1-6|8-9]|9[0-4|6-9])\d{7})"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        return None

    def find_address(self, text: str):
        """Extract address from text."""
        address_keywords = ['Địa chỉ', 'Address', 'DC:', 'Đ/c:', 'Chi nhánh', 'Văn phòng', 'Trụ sở', 'Khu vực', 'TP.', 'Q.']
        location_words = ['Phường', 'Quận', 'Huyện', 'Tỉnh', 'Thành phố', 'Đường', 'Phố', 'Ngõ', 'Khu', 'Số']
        exclude_keywords = ['tổng', 'total', 'ngày', 'date', 'số lượng', 'giá', 'mã']

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            for keyword in address_keywords:
                if line.lower().startswith(keyword.lower()):
                    address = line[len(keyword):].strip(':').strip()
                    if len(address) > 10:
                        return address

        for line in lines:
            line = line.strip()
            if len(line) < 10 or len(line) > 100:
                continue
            word_count = sum(1 for word in location_words if word.lower() in line.lower())
            if word_count >= 2 and re.search(r'\d+', line) and ',' in line:
                if not any(keyword.lower() in line.lower() for keyword in exclude_keywords):
                    return line
        return None

    def clean_amount(self, value: str) -> str:
        """Clean amount string by handling decimal formatting."""
        return value.replace('.', '').replace(',', '.')

    def find_total(self, text: str):
        """Extract total amount from text."""
        patterns = [
            r"Tổng cộng[:\s]*([\d.,]+)",
            r"Tổng tiền[:\s]*([\d.,]+)",
            r"Tổng số tiền[:\s]*([\d.,]+)",
            r"Tổng thanh toán[:\s]*([\d.,]+)",
            r"Tổng[:\s]*([\d.,]+)",
            r"Total[:\s]*([\d.,]+)",
            r"Thành tiền[:\s]*([\d.,]+)",
            r"([\d.,]+)\s*(?:VND|₫|đ)"
        ]
        fuzzy_patterns = [
            r"T[oôơ0]ng\s*t[iíìîï]e[nm][: ]*([\d.,]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self.clean_amount(match.group(1))
        for pattern in fuzzy_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self.clean_amount(match.group(1))
        return None

    def find_date(self, text):
        """Extract date from text."""
        patterns = [
            r"Ngày[:\s]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
            r"Date[:\s]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
            r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

