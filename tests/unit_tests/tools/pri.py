import re
import json
import os
from playwright.sync_api import sync_playwright
from langchain.llms import OpenAI, OpenAIChat
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from urllib.parse import urljoin
from typing import List


def count_tokens(text):
    """A simple approximation of token count similar to OpenAI's GPT-3."""
    words = re.findall(r'\b\w+\b', text)
    spaces = re.findall(r'\s', text)
    others = re.findall(r'\W', text)
    return len(words) + len(spaces) + len(others)


def limit_tokens(text: str, maxToken=2000):
    tokens = text.split()
    res = ' '.join(tokens[:maxToken])
    print(
        f"limit token before {count_tokens(text)} ====> after {count_tokens(res)}")
    return res


def getHtmlContent(url, timeout=10 * 1000):
    with sync_playwright() as play_wright:
        try:
            browser = play_wright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url=url, timeout=timeout)
            return page.inner_html('body')
        except Exception as e:
            print(
                f"get error while getting content from {url} and the error is {e}")
            return page.inner_html('body')
        finally:
            page.close()
            context.close()
            browser.close()


def extractPtext(htmlContent):
    soup = BeautifulSoup(htmlContent, "html.parser")
    paragraphs = soup.find_all('p')
    return [p.get_text() for p in paragraphs]


def extractLinks(htmlContent, base):
    soup = BeautifulSoup(htmlContent, 'html.parser')
    links = soup.find_all('a')
    # Extract href attribute from each link
    urls = [urljoin(base, link.get('href', "")) for link in links]
    return urls


def searchDuckDuckGoforLinks(keyword, llm, num=10):
    """search for a list of links from search engine"""
    with sync_playwright() as play_wright:
        browser = play_wright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f'https://duckduckgo.com/?q={keyword}&t=h_&ia=web')
        body_inner_text = page.inner_html('body')
        soup = BeautifulSoup(body_inner_text, "lxml")
        base = page.url
        anchors = soup.find_all("a", limit=None)
        l = [urljoin(base, a.get("href", "")) for a in anchors]
        filterRegex = re.compile(r'(javascript|duckduckgo)')
        filtered = list(
            filter(lambda string: not filterRegex.search(string), l))
        p = f"pick {num} urls you think are most related to \"{keyword}\" from the following url list \n ````` \n {json.dumps(filtered)}\n ````` \n please respond in JSON format, for example {json.dumps(['url1', 'url2'])} \n respond fast please"
        res = llm([HumanMessage(content=p)])
        page.close()
        context.close()
        browser.close()
        return json.loads(res.content)


def sumSinglePage(url: str, llm: ChatOpenAI, query: str):
    try:
        print(f"start summarize page {url}")
        body_inner_text = getHtmlContent(url)
        allText = extractPtext(body_inner_text)
        theText = ' '.join(allText)
        if (count_tokens(theText) > 4000):
            theText = limit_tokens(theText, 3000)
        # print(theText)
        p = f"Got a query from user as \"{query}\" give me an informative and short summary(should includes more facts like date, location, people's names and so on) of the following text \n ``` \n {theText} \n ``` \n"
        # if there are too many tokens in the text?
        print(f"this time the token count is {count_tokens(p)}")
        res = llm([HumanMessage(content=p)])
        return res.content
    except Exception as e:
        print(f"get error while sum page {url} ====>", e)
        return ''


def askKeyWords(query: str) -> list[str]:
    llm = ChatOpenAI()
    num = 5
    p = f"I want to search about \"{query}\", could you suggest me {num} potential keyword for search engine(You can make some rather specific keywords rather than general ones)? respond me in JSON format, for example {json.dumps(['keyword1', 'keyword2'])}"
    print(p)
    res = llm([HumanMessage(content=p)])
    return json.loads(res.content)


tools = []

#
class askKeywordInput(BaseModel):
    query: str = Field(description="the search query from user")
    llm: ChatOpenAI = Field(description="llmchat model used by tool")
    limit: int = Field(description="number of key words generated, the default value is 5")

def d():
    return 1

tools.append(
    StructuredTool.from_function(
        func=askKeyWords,
        name="askForKeywods",
        description="useful when you need to generate a series of related keyword based on user's query",
        # args_schema=askKeywordInput
    )
)


# class getLinksFromSearchEngineInput(BaseModel):
#     keyword: str = Field(description="the keyword for searching by searchEngine")
#     llm: ChatOpenAI = Field(description="llmchat model used by tool")
#     num: int = Field(description="number of links to be extracted, default value is 10")
#
#
# tools.append(
#     StructuredTool.from_function(
#         func=searchDuckDuckGoforLinks,
#         name="getLinkFromSearchEngine",
#         description="useful when you need to find some link by a keyword from search engine",
#         args_schema=getLinksFromSearchEngineInput
#     )
# )
#
#
# class sumSinglePageInput(BaseModel):
#     url: str = Field(description="the url of the web page")
#     llm: ChatOpenAI = Field(description="llmchat model used by tool")
#     query: str = Field(description="the query from user based on which the summary should be generated")
#
#
# tools.append(
#     StructuredTool.from_function(
#         func=sumSinglePage,
#         name="getSummaryFromWebpage",
#         description="useful when you want to get summary from a page by url",
#         args_schema=sumSinglePageInput
#     )
# )
os.environ["GOOGLE_CSE_ID"] = "834a7da892d244bb7"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAiWw_5tqSZ1nWq-z1ygOcO099cqjPAK6Y"
os.environ["OPENAI_API_KEY"] = "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"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
query = "find me some info about syria war"
agent.run(
    f"here is a question from user as \"{query}\", get some keyword based on this query, for each query search for some link by search engine, and get summary from each link, finally based on the list of summaries generate some info for the user")
