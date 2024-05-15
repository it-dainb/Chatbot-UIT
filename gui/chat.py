from taipy.gui import notify, navigate, Markdown

chat_md = Markdown("""
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# UIT **Chat**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=new_chat|>
<|Log Out|button|class_name=fullwidth plain|id=log_out|on_action=user_log_out|>
### Previous activities ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>
<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|id=chat_table|rebuild|>
<|part|class_name=card p0 pl2 pr3 pt1 pb1|
<|{current_user_message}|input|label=Hãy điền câu hỏi vào đây...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
""")

DEFAULT_MESSAGE = "Xin Chào, tôi là Chatbot tuyển sinh của UIT. Tôi có thể giúp gì cho bạn?"

def user_log_out(state):
    state.token = None
    state.login_username = ""
    state.login_password = ""

    state.conversation = {
        "Conversation": ["Xin Chào, tôi là Chatbot tuyển sinh của UIT. Tôi có thể giúp gì cho bạn?"]
    }

    state.current_user_message = ""
    state.chat_id = ""

    state.past_conversations = []
    state.conversation = {
        "Conversation": [DEFAULT_MESSAGE]
    }
    state.selected_conv = None
    state.selected_row = [1]

    notify(state, "success", "Logged out successfully!")

    navigate(state, "Login")