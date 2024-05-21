# end of taipy. gizmo. uis - chat. jpg ( tinym
# -*- coding: utf-8 -*-
from taipy.gui import Gui, Markdown

# Tạo phần tử Markdown cho chữ "Welcome"
welcome_md = Markdown("""
<style>
    .uit-chat {
        font-family: 'Arial', sans-serif; /* Thay đổi font chữ theo yêu cầu */
        color: #FFFFFF; /* Màu chữ trắng */
        padding: 10px; /* Khoảng cách giữa chữ và viền */
        border-radius: 5px; /* Bo tròn viền */
        text-align: center; /* Căn giữa ngang */
        font-size: 100px; /* Kích thước chữ */
        font-weight: bold; /* Đậm chữ */
        margin-top: 50px;
    }
    .uit {
        color: #FF5733; /* Màu chữ UIT */
    }
</style>

<div class="uit-chat">
    <span class="uit">UIT</span> Chat
</div>
<div class="uit-chat">
    <span class="uit">Sẵn sàng</span> để <span class="uit">phục vụ</span> bạn!
</div>
""")