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
    """
     @brief Reset the state to default values. This is called when user clicks a button in the conversation list to reset the state to the default values
     @param state The state to reset
    """
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
     @brief Called when the app is initialized. This is a callback function that will be called by : func : ` pytorch_app. init ` in order to initialize the app.
     @param state The state of the app. See : func : ` pytorch_app. init ` for more information.
     @return None or an error object if something went wrong. The error object will be a subclass of : class : ` pants. core. errors. PreconditionFailure `
    """
    """
    Initialize the app.

    Args:
        - state: The current state of the app.
    """
    reset(state)

def update_context(state: State) -> None:
    """
     @brief Update the context with the user's message and the AI. This is used to send messages to the user and get access token
     @param state The state of the app
     @return The token that was sent to the user and the AI if successful None if not ( login_open
    """
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

    # This function is called by the user to login the user.
    if response.status_code != 200:
        # login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login login
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
     @brief Send the user's message to the API and update the context. This is the function that should be called in the context of the app
     @param state The state of the
    """
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

    # If there are no past conversations in the chat_id.
    if len(state.past_conversations) == 0 or state.chat_id != chat_id:
        past_conv = state.past_conversations.copy()
        past_conv = [[len(state.past_conversations), chat_id, state.conversation]] + past_conv

        state.past_conversations = past_conv
    
    state.chat_id = chat_id

def style_conv(state: State, idx: int, row: int) -> str:
    """
     @brief Apply a style to the conversation table depending on the message. This is used to display the message in the conversation table
     @param state The current state of the app
     @param idx The index of the message in the table.
     @param row The row of the message in the table.
     @return The style to apply to the message or None if there is no message in the table for the given
    """
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    # Return the message type for the given index.
    if idx is None:
        return None
    elif idx % 2 == 1:
        return "user_message"
    else:
        return "gpt_message"


def on_exception(state, function_name: str, ex: Exception) -> None:
    """
     @brief Catches exceptions and notifies user in Taipy GUI. This function is called when an exception is raised in a function.
     @param state State to use for notification. Can be any object that implements
     @param function_name Name of function where exception occured
     @param ex Exception that was raised.
    """
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
     @brief Reset the chat by clearing the conversation. This is useful for testing purposes. If you want to make a new chat use : func : ` ~chats. Bot. new_chat ` instead.
     @param state The state of the app. Must be a : class : ` ~chats. State ` object.
     @return The return value of : func : ` ~chats. Bot. new_chat `. >>> state = state. get_current_state () Traceback ( most recent call last ) : NoState
    """
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


def tree_adapter(item: list):
    """
     @brief Adapter for tree_display to return id and displayed string. This adapter is used to convert past_conversations to id and displayed string
     @param item element of past_conversations to be converted
     @return tuple of id and displayed string if there is more than one conversation otherwise ( id displayed string ). In case of empty conversation it will return " Empty conversation
    """
    """
    Converts element of past_conversations to id and displayed string

    Args:
        item: element of past_conversations

    Returns:
        id and displayed string
    """
    identifier = str(item[0])
    # Conversation identifier conversation 1 50 50
    if len(item[2]["Conversation"]) > 2:
        return (identifier, item[2]["Conversation"][-1][:50] + "...")
    return (identifier, "Empty conversation")


def select_conv(state: State, var_name: str, value) -> None:
    """
     @brief Selects the conversation from past conversations. This is a callback for state changes that need to be performed after the user presses the select button.
     @param state The state of the app. It is passed by the : class : `. Context ` object as well as a variable called ` state ` to allow
     @param var_name
     @param value
    """
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
    """
     @brief Handle menu actions. Navigates to the page specified in the payload and calls the appropriate action depending on the page.
     @param state The state to be used in the navigation.
     @param action The action to be performed. This is passed to : func : ` waflib. actions. Action. menu_action ` as it is called from
     @param payload The payload containing the page to navigate to.
     @return The return value of : func : ` waflib. actions. Action. navigate ` after setting the state '
    """
    page = payload["args"][0]

    state.login_open = False
    # Login page. If login is successful login will be redirected to the login page.
    if page == "Login":
        # If the user is logged in login open.
        if state.token is None:
            state.login_open = True
        else:
            notify(state, "info", "You are already logged in! Please logout first!")
            navigate(state, "Chat")
            return

    # Register page for register page.
    if page == "Register":
        state.register_open = True

        state.register_username = ""
        state.register_password = ""
        state.confirm_password = ""

    # If the page is Chat then login first.
    if page == "Chat" and state.token is None:
        notify(state, "info", "Please login first!")
        state.login_open = True
        navigate(state, "Login")
        return
    
    navigate(state, page)

def navigate_to_home(state):
    """
     @brief Navigate to the Home page. This is a convenience function to make it easier to navigate to the Home page without having to worry about opening the login and registration pages in the middle of the navigation
     @param state The state of the
    """
    state.login_open = True
    state.register_open = True
    navigate(state, "Home")

# This function is called by the main module.
if __name__ == "__main__":

    gui = Gui(pages=pages)
    gui.run(use_reloader=True, debug=True, dark_mode=True, title="💬 UIT Chat")
