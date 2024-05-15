# -*- coding: utf-8 -*-
from taipy.gui import notify, Markdown, navigate
import requests

login_md = Markdown("""
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
<|{login_open}|dialog|title=Login|on_action=navigate_to_home|width=30%|

**Username**
<|{login_username}|input|label=Username|class_name=fullwidth|>

**Password**
<|{login_password}|input|password|label=Password|class_name=fullwidth|>

<br/>
<|Login|button|class_name=fullwidth plain|on_action=login_user|>
<|Register|button|class_name=fullwidth plain|on_action=navigate_to_register|>

|>
""")

def navigate_to_register(state):
    state.register_open = True
    navigate(state, "Register")

def login_user(state):
    # Put your own authentication system here
    response = requests.post(
        "http://localhost:8080/user/login",
        data = {
            "username": state.login_username,
            "password": state.login_password
        }
    )

    if response.status_code != 200:
        notify(state, "error", response.json()["detail"])
        return

    response = response.json()

    print(response)
    
    state.token = response['token_type'].capitalize() + ' ' + response["access_token"]
    state.login_open = False

    navigate(state, "Chat")
    
    notify(state, "success", "Login successful!")