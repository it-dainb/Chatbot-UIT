from taipy.gui import Gui, navigate, State, notify

from gui.chat import chat_md, DEFAULT_MESSAGE
from gui.login import login_md
from gui.register import register_md
from gui.welcome import welcome_md

import requests

pages = {
    "/": "<|menu|lov={page_names}|on_action=menu_action|color=red|>",
    "Home": welcome_md,
    "Login": login_md,
    "Register": register_md,
    "Chat": chat_md,
}

page_names = [page for page in pages.keys() if page != "/"]

token = None

conversation = {
    "Conversation": [DEFAULT_MESSAGE]
}

current_user_message = ""
chat_id = ""

past_conversations = []

selected_conv = None
selected_row = [1]

login_open = True
register_open = True

login_username = ""
login_password = ""

register_username = ""
register_password = ""
confirm_password = ""

def reset(state):
    state.token = None
    state.conversation = {
        "Conversation": [DEFAULT_MESSAGE]
    }

    state.current_user_message = ""
    state.chat_id = ""

    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]

def on_init(state: State) -> None:
    """
    Initialize the app.

    Args:
        - state: The current state of the app.
    """
    reset(state)

def update_context(state: State) -> None:
    """
    Update the context with the user's message and the AI's response.

    Args:
        - state: The current state of the app.
    """
    response = requests.post(
        "http://localhost:8080/engine/chat",
        data = {
            "text": state.current_user_message,
            "chat_id": state.chat_id
        },
        headers = {
            "Authorization": state.token
        }
    )

    if response.status_code != 200:
        if state.login_username == "" or state.login_password == "":
            notify(state, "info", "Please login first!")

            state.login_open = True
            reset(state)
            
            navigate(state, "Login")
            return

        response = requests.post(
            "http://localhost:8080/user/login",
            data = {
                "username": state.login_username,
                "password": state.login_password
            }
        )

        response = response.json()
        state.token = response['token_type'].capitalize() + ' ' + response["access_token"]

        response = requests.post(
            "http://localhost:8080/engine/chat",
            data = {
                "text": state.current_user_message,
                "chat_id": state.chat_id
            },
            headers = {
                "Authorization": state.token
            }
        )

    response = response.json()
    
    state.selected_row = [len(state.conversation["Conversation"]) + 1]

    return response["response"], response["chat_id"]

def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the context.

    Args:
        - state: The current state of the app.
    """
    notify(state, "info", "Sending message...")

    answer, chat_id = update_context(state)
    
    conv = state.conversation._dict.copy()

    conv["Conversation"] += [state.current_user_message, answer]

    state.current_user_message = ""
    state.conversation = conv

    notify(state, "success", "Response received!")

    if len(state.past_conversations) == 0 or state.chat_id != chat_id:
        past_conv = state.past_conversations.copy()
        past_conv = [[len(state.past_conversations), chat_id, state.conversation]] + past_conv

        state.past_conversations = past_conv
    
    state.chat_id = chat_id

def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 1:
        return "user_message"
    else:
        return "gpt_message"


def on_exception(state, function_name: str, ex: Exception) -> None:
    """
    Catches exceptions and notifies user in Taipy GUI

    Args:
        state (State): Taipy GUI state
        function_name (str): Name of function where exception occured
        ex (Exception): Exception
    """
    notify(state, "error", f"An error occured in {function_name}: {ex}")


def new_chat(state: State) -> None:
    """
    Reset the chat by clearing the conversation.

    Args:
        - state: The current state of the app.
    """
    state.conversation = {
        "Conversation": [DEFAULT_MESSAGE]
    }
    state.chat_id = ""
    state.current_user_message = ""


def tree_adapter(item: list) -> [str, str]:
    """
    Converts element of past_conversations to id and displayed string

    Args:
        item: element of past_conversations

    Returns:
        id and displayed string
    """
    identifier = str(item[0])
    if len(item[2]["Conversation"]) > 2:
        return (identifier, item[2]["Conversation"][-1][:50] + "...")
    return (identifier, "Empty conversation")


def select_conv(state: State, var_name: str, value) -> None:
    """
    Selects conversation from past_conversations

    Args:
        state: The current state of the app.
        var_name: "selected_conv"
        value: [[id, conversation]]
    """
    _, chat_id, conversation = state.past_conversations[len(state.past_conversations) - 1 - value[0][0]]

    state.conversation = conversation
    state.chat_id = chat_id
    
    state.selected_row = [len(state.conversation["Conversation"]) - 1]

past_prompts = []
def menu_action(state, action, payload):
    page = payload["args"][0]

    state.login_open = False
    if page == "Login":
        if state.token is None:
            state.login_open = True
        else:
            notify(state, "info", "You are already logged in! Please logout first!")
            navigate(state, "Chat")
            return

    if page == "Register":
        state.register_open = True

        state.register_username = ""
        state.register_password = ""
        state.confirm_password = ""

    if page == "Chat" and state.token is None:
        notify(state, "info", "Please login first!")
        state.login_open = True
        navigate(state, "Login")
        return
    
    navigate(state, page)

def navigate_to_home(state):
    state.login_open = True
    state.register_open = True
    navigate(state, "Home")

if __name__ == "__main__":

    gui = Gui(pages=pages)
    gui.run(use_reloader=True, debug=True, dark_mode=True, title="💬 UIT Chat")
