import pandas as pd
from typing import List, Dict
from core.utils.preprocessing import clean_text

from core.utils.logger import logger

ANSWER_MAP = {
    "hỏi_lại_khi_không_hiểu|hỏi_lại": {
        "Answer":["UIT Bot không hiểu nội dung câu hỏi của bạn. Mong bạn vui lòng nhập câu hỏi rõ ràng hơn"],
        'Description': 'hỏi lại',
        'Product': 'Khác'
    },
    "truy_vấn_ngoài_phạm_vi|phạm_vi": {
        "Answer": ["Câu hỏi này ngoài phạm vi trả lời của UIT Bot ạ"],
        'Description': 'phạm vi',
        'Product': 'Khác'
    },
    "lời_chào_kết_thúc|kết_thúc": {
        "Answer": ["Cảm ơn bạn đã sử dụng UIT Bot. Rất vui khi hỗ trợ cho bạn"],
        'Description': 'phạm vi',
        'Product': 'Khác'
    },
    "hỏi_đáp_không_ý_nghĩa|hỏi_đáp": {
        "Answer": ["Bạn có cần UIT Bot hỗ trợ giải đáp vấn đề gì nữa không?"],
        'Description': 'hỏi đáp',
        'Product': 'Khác'
    },
    "lời_chào_mở_đầu|mở_đầu": {
        "Answer": ["Xin chào, UIT bot có thể hỗ trợ gì cho bạn!"],
        'Description': 'mở đầu',
        'Product': 'Khác'
    },
    "chuyển_hội_thoại_cho_agent|agent": {
        "Answer": ["Bạn có thể liên hệ trực tiếp thông tin tư vấn của UIT Tuyển sinh để được giải đáp trực tiếp"],
        'Description': 'agent',
        'Product': 'Khác'
    },
    "đề_xuất_khi_không_xác_định|đề_xuất": {
        "Answer": ["Xin lỗi không thể trả lời câu hỏi do sự nhập nhằng trong lựa chọn câu trả lời nên chúng tôi đề xuất các câu trả lời như sau:"],
        'Description': 'đề xuất',
        'Product': 'Khác'
    }
}

class Database:
    def __init__(self, path_indomain: str, path_outdomain: str):
        self.path_indomain = path_indomain
        self.path_outdomain = path_outdomain
        
        self.read_file()
        
    def read_file(self) -> None:
        logger.info("Reading file")
        
        self.synonyms_dictionary = self.read_share_knowledge()
        self.answer_database, self.answer_template = self.read_answer_database()
        self.products = self.read_product(self.answer_database, self.answer_template)
        self.question = pd.read_excel(self.path_indomain, sheet_name="Question", skiprows=2)

        self.indomain = None
        
    def read_share_knowledge(self) -> None:
        logger.info("Reading share knowledge")
        
        self.share = pd.read_excel(self.path_indomain, sheet_name="Share Knowledge", skiprows=2)
        keyword = self.share["Keyword"].values.tolist()
        synonyms = self.share["Synonym"].values.tolist()
        
        synonyms_dictionary = dict(zip(keyword, synonyms))
        return synonyms_dictionary

    def read_answer_database(self) -> None:
        logger.info("Reading answer database")
        
        self.answer = pd.read_excel(self.path_indomain, sheet_name="Answer", skiprows=2)
        self.intent = pd.read_excel(self.path_indomain, sheet_name="Intent", skiprows=2)
        
        answer_expand = \
        self.answer.set_index('Pattern Template')\
        .join(
            self.intent.set_index('Pattern Template'),
            rsuffix='_right'
        )
        
        answer_expand["Pattern Template"] = answer_expand.index
        answer_expand = answer_expand[["Answer", "Pattern Template", "Description", "Product"]]
                
        database_raw = answer_expand.copy()
        database = database_raw.set_index('Pattern Template').T.to_dict()

        # Check again to update multiple answer
        for key, value in database.items():
            rows = database_raw[database_raw['Pattern Template'] == key]
            rows_answer = rows['Answer'].to_list()
            value["Answer"] = rows_answer
            
        answer_template = [str(item) for item in database.keys()]

        for intent, answer in ANSWER_MAP.items():
            if intent not in database or (intent in database and 'Answer' not in database[intent]):
                database[intent] = answer

        return database, answer_template

    def read_product(self, answer_database: Dict, answer_template: List[str]) -> None:
        logger.info("Reading product")
        
        list_product = []
        for template in answer_template:
            if template not in answer_database:
                continue
            
            if "Product" not in answer_database[template]:
                continue

            if str(answer_database[template]["Product"]).strip() in ["", "Khác"]:
                continue

            list_product.append(answer_database[template]["Product"])

        return set(list_product)

    def create_train_label_data(self) -> Dict:
        logger.info("Reading in domain")
        self.indomain = pd.read_excel(self.path_indomain, sheet_name="Question", skiprows=2)
        self.indomain['Question'] = self.indomain['Question'].apply(lambda x: clean_text(x, self.synonyms_dictionary))
        
        logger.debug("Creating train label data")
        
        X = self.indomain["Question"].tolist()
        y = self.indomain["Pattern Template"].tolist()

        list_label = []
        for item in set(y):
            label = item.split("|")[0].strip()
            
            if label.strip() in ["None", "", "truy_vấn_ngoài_phạm_vi"]:
                continue

            if label.strip() in list_label:
                continue

            list_label.append(label)

        X_data = []
        y_data = []
        for intents, x in zip(y, X):
            label = intents.split("|")[0].strip()
            if label in list_label:
                X_data.append(x)
                y_data.append(list_label.index(label))

        num_class = len(list_label)
        max_length = max([len(s.split()) for s in X_data])
        
        return {
            'x': X_data,
            'y': y_data,
            'num_class': num_class,
            'max_length': max_length,
            'class': list_label
        }

    def create_train_in_out_data(self) -> Dict:
        if self.indomain is None:
            logger.info("Reading in domain")
            self.indomain = pd.read_excel(self.path_indomain, sheet_name="Question", skiprows=2)
            self.indomain['Question'] = self.indomain['Question'].apply(lambda x: clean_text(x, self.synonyms_dictionary))
            
        logger.info("Reading out domain")   
        self.outdomain = pd.read_excel(self.path_outdomain, sheet_name="Full out of domain", skiprows=2)
        self.outdomain['Question'] = self.outdomain['Question'].apply(lambda x: clean_text(x, self.synonyms_dictionary))

        logger.debug("Creating train in out data")
        
        X_in_data = self.indomain["Question"].tolist()
        X_out_data = self.outdomain["Question"].tolist()

        list_label = ['in', 'out']
        
        X_data = X_in_data + X_out_data
        y_data = [0] * len(X_in_data) + [1] * len(X_out_data)

        num_class = 2
        max_length = max([len(s.split()) for s in X_data])
        
        return {
            'x': X_data,
            'y': y_data,
            'num_class': num_class,
            'max_length': max_length,
            'class': list_label
        }