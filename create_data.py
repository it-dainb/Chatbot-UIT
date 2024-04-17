def select(input_list, type = "full"):
    
    # print()
    for idx, intent in enumerate(input_list):
        intent = intent.replace("_", " ")
        intent = short_hand.get(intent, intent.capitalize())

        prefix = " "*(len(str(len(input_list))) - len(str(idx + 1)))

        # print(f"    {prefix}{idx+1}. {intent}")

    # print()
    select = input_list[int(input("Nhập lựa chọn: ")) - 1]
    
    return select

short_hand = {
    'ktpm': 'kỹ thuật phần mềm',
    'cntt': 'công nghệ thông tin',
    'clc': 'chất lượng cao',
    'việt_nhật': 'việt nhật',
    'dgnl': 'đánh giá năng lực',
    'attt': 'an toàn thông tin',
    'httt': 'hệ thống thông tin',
    'ktmt': 'kỹ thuật máy tính',
    'mmt': 'mạng máy tính',
    'tiên_tiến': 'tiến tiến',
    'tmdt': 'thương mại điện tử',
    'iot': 'hệ thống nhúng và IOT',
    'khmt': 'khoa học máy tính',
    'khdl': 'khoa học dữ liệu',
    'ttnt': 'trí tuệ nhân tạo',
    'thpt': 'trung học phổ thông'
}

intent_list = ["hỏi_đáp_điểm_chuẩn","hỏi_đáp_ngành","hỏi_đáp_xét_tuyển","thông_tin_chỉ_tiêu","hỏi_đáp_ktx","hỏi_đáp_xe_bus","hỏi_đáp_uit","hỏi_đáp_tổ_hợp","hỏi_đáp_nghề_nghiệp","thông_tin_học_bổng"]
intent = select(intent_list)
# print(intent)
# print()

# print()
year = "năm_" + input("Nhập năm: ")
# print()

type_list = ["dgnl", "thpt"]
type = select(type_list)
# print(type)
# print()

special = {
    'clc': ["attt", 'httt', 'khmt', 'ktmt', 'ktpm', 'mmt', 'tmdt'],
    'việt_nhật': ['cntt'],
    'tiên_tiến': ['httt'],
    'ttnt': ['khmt'],
    'iot': ['ktmt']
}

data = {}
sub_list = "ktpm,cntt,attt,httt,ktmt,mmt,tmdt,khmt,khdl".split(",")
for sub in sub_list:
    temp = [intent, type, sub]
    # print()
    # # print(short_hand[sub].capitalize())
    
    key_temp = []
    for key, value in special.items():
        if int(year.split("_")[-1]) >= 2022 and key == "clc":
            continue
        
        if sub in value:
            key_temp.append(key)

    for spe_key in key_temp:
        key = "|".join(temp + [spe_key] + [year])
        # print(key)
        data[key] = input("Nhập thông tin: ")
        # print()

    key = "|".join(temp + [year])
    # print(key)
    data[key] = input("Nhập thông tin: ")
    # print()

    
for key, value in data.items():
    prefix = []
    if 'hỏi_đáp_điểm_chuẩn' in key:
        prefix.append('Điểm chuẩn')
    
    is_dgnl = "dgnl" in key
    if "dgnl" in key:
        prefix.append('ĐGNL')
        prefix.append(year)
    
    for sub in sub_list:
        if sub in key:
            prefix.append("ngành")
            prefix.append(short_hand[sub].capitalize())
            if not is_dgnl:
                prefix.append(year)
            break
    
    for spe in special.keys():
        if spe in key:
            prefix.append("hướng")
            prefix.append(short_hand[spe].capitalize())
            break
    
    prefix.append("là")
    prefix.append(value)

    data[key] = " ".join(prefix).replace("_", " ")

# for value in data.values():
    # print(value)

# print()

# for value in data.keys():
    # print(value)