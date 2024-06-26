# -*- coding: utf-8 -*-
from taipy.gui import notify, Markdown, navigate
import requests

register_md = Markdown("""
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
<|{register_open}|dialog|title=Register|width=30%|on_action=navigate_to_home|

**Username**
<|{register_username}|input|label=Username|class_name=fullwidth|>

**Password**
<|{register_password}|input|password|label=Password|class_name=fullwidth|>

**Confirm Password**
<|{confirm_password}|input|password|label=Confirm Password|class_name=fullwidth|>

<br/>
<|Register|button|class_name=fullwidth plain|on_action=register_account|>
|>
""")

def register_account(state):
    if state.register_password == state.confirm_password:

        response = requests.post(
            "http://localhost:8080/user/register",
            data={
                "username": state.register_username,
                "password": state.register_password
            }
        )

        if response.status_code != 200:
            notify(state, "error", response.json()["detail"])

            state.register_username = ""
            state.register_password = ""
            state.confirm_password = ""
            
            return

        response = response.json()
        state.token = response['token_type'].capitalize() + ' ' + response["access_token"]
        
        notify(state, "success", "Account registered successfully!")

        state.login_username = state.register_username
        state.login_password = state.register_password
        
        state.register_open = False

        navigate(state, "Chat")
    else:
        notify(state, "error", "Passwords do not match!")

        state.confirm_password = ""
        
def cancel(state):
    state.register_open = False