import joblib
import string
from core.utils.preprocessing import clean_text
import unicodedata

data = joblib.load("Data/5_ngrams.pkl")

accented_chars_vietnamese = [
            'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
            'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
            'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
            'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
            'í', 'ì', 'ỉ', 'ĩ', 'ị',
            'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
            'đ',
        ]
accented_chars_vietnamese.extend([c.upper() for c in accented_chars_vietnamese])
pad_token = '\x00'
alphabet = list((f'{pad_token} _' + string.ascii_letters + string.digits + ''.join(accented_chars_vietnamese)))

for i in range(len(data)):
    for c in data[i]:
        if c not in alphabet:
            print(c, " | ", data[i])
            print(clean_text(data[i], {}, tokenizer=False))
            print(unicodedata.normalize("NFC", data[i]))
            print()
            
        
