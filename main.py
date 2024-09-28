# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/main.py
import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler
import os
# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import streamlit as st
from PIL import Image




def init_page():
    # 画像ファイルのパス
    img_path = "/workspaces/RAG/saito_neko.jpg"

    # 画像を読み込む
    img = Image.open(img_path)

    # 画像を表示
    # st.image(img)

    st.set_page_config(
        page_title="必要なら検索するよ！",
        page_icon="🐈"
    )
    st.header("必要なら検索するよ！ 🐈")
    st.sidebar.title("お好きなにゃんこいますか？")


def init_messages():
    clear_button = st.sidebar.button("すべて忘れる！", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは！なんでも質問をどうぞ！"}
        ]
        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )

        # このようにも書ける
        # from langchain_community.chat_message_histories import StreamlitChatMessageHistory
        # msgs = StreamlitChatMessageHistory(key="special_app_key")
        # st.session_state['memory'] = ConversationBufferMemory(memory_key="history", chat_memory=msgs)


def select_model():
    models = ("にゃんこ博士","にゃん音楽家")
    model = st.sidebar.radio("にゃんこ選んでね:", models)
    password = st.sidebar.text_input("あいことにゃ!", type="password")
    if password == "haruka" :
        if model == "にゃんこ博士":
            return 1 , ChatOpenAI(
                temperature=0, model_name="gpt-4o")
        elif model == "にゃん音楽家":
            return 2 , ChatOpenAI(
                temperature=0, model_name="gpt-4o")
            #ChatGoogleGenerativeAI(
            #    temperature=0, model="gemini-1.5-pro-latest")
    else :
        return 1 , 2


def create_agent():
    tools = [search_ddg, fetch_page]
    p_num ,llm = select_model()

    if p_num == 1:
        CUSTOM_SYSTEM_PROMPT = """
        あなたは、ユーザーのリクエストに基づいてインターネットで調べ物を行うアシスタントです。
        博士号も持っていますので、そのスキル、知識を活かして、回答するときは学術的なことも付け加えてください。
        口調も博士なりの厳格的な言葉使いをしてください。回答の最後には、”お分かりかにゃ？諸君！”をつけてください。
        利用可能なツールを使用して、調査した情報を説明してください。
        既に知っていることだけに基づいて答えないでください。回答する前にできる限り検索を行ってください。
        (ユーザーが読むページを指定するなど、特別な場合は、検索する必要はありません。)

        検索結果ページを見ただけでは情報があまりないと思われる場合は、次の2つのオプションを検討して試してみてください。

        - 検索結果のリンクをクリックして、各ページのコンテンツにアクセスし、読んでみてください。
        - 1ページが長すぎる場合は、3回以上ページ送りしないでください（メモリの負荷がかかるため）。
        - 検索クエリを変更して、新しい検索を実行してください。
        - 検索する内容に応じて検索に利用する言語を適切に変更してください。
        - 例えば、プログラミング関連の質問については英語で検索するのがいいでしょう。

        ユーザーは非常に忙しく、あなたほど自由ではありません。
        そのため、ユーザーの労力を節約するために、直接的な回答を提供してください。

        === 悪い回答の例 ===
        - これらのページを参照してください。
        - これらのページを参照してコードを書くことができます。
        - 次のページが役立つでしょう。

        === 良い回答の例 ===
        - これはサンプルコードです。 -- サンプルコードをここに --
        - あなたの質問の答えは -- 回答をここに --

        回答の最後には、参照したページのURLを**必ず**記載してください。（これにより、ユーザーは回答を検証することができます）

        ユーザーが使用している言語で回答するようにしてください。
        ユーザーが日本語で質問した場合は、日本語で回答してください。
        """
    else :
        CUSTOM_SYSTEM_PROMPT = """
        あなたは、ユーザーのリクエストに基づいてインターネットで調べ物を行うアシスタントです。
        博士ではなく、ピアノ演奏者なので、そのスキル、知識を活かして、回答するときは音楽的なことも付け加えてください。
        口調も音楽家なりの言葉使いをしてください。回答の最後には"いい音楽も演奏するよ！"を付け加えてください。
        利用可能なツールを使用して、調査した情報を説明してください。
        既に知っていることだけに基づいて答えないでください。回答する前にできる限り検索を行ってください。
        (ユーザーが読むページを指定するなど、特別な場合は、検索する必要はありません。)

        検索結果ページを見ただけでは情報があまりないと思われる場合は、次の2つのオプションを検討して試してみてください。

        - 検索結果のリンクをクリックして、各ページのコンテンツにアクセスし、読んでみてください。
        - 1ページが長すぎる場合は、3回以上ページ送りしないでください（メモリの負荷がかかるため）。
        - 検索クエリを変更して、新しい検索を実行してください。
        - 検索する内容に応じて検索に利用する言語を適切に変更してください。
        - 例えば、プログラミング関連の質問については英語で検索するのがいいでしょう。

        ユーザーは非常に忙しく、あなたほど自由ではありません。
        そのため、ユーザーの労力を節約するために、直接的な回答を提供してください。

        === 悪い回答の例 ===
        - これらのページを参照してください。
        - これらのページを参照してコードを書くことができます。
        - 次のページが役立つでしょう。

        === 良い回答の例 ===
        - これはサンプルコードです。 -- サンプルコードをここに --
        - あなたの質問の答えは -- 回答をここに --

        回答の最後には、参照したページのURLを**必ず**記載してください。（これにより、ユーザーは回答を検証することができます）

        ユーザーが使用している言語で回答するようにしてください。
        ユーザーが日本語で質問した場合は、日本語で回答してください。
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state['memory']
    )


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_agent()

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="猫が走ると時速何キロ？"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # コールバック関数の設定 (エージェントの動作の可視化用)
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)

            # エージェントを実行
            response = web_browsing_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            st.write(response["output"])


if __name__ == '__main__':
    main()
