"""Agent that interacts with OpenAPI APIs via a hierarchical planning approach."""
import json
import re
from typing import List, Optional

import yaml

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.openapi.planner_prompt import (
    API_CONTROLLER_PROMPT,
    API_CONTROLLER_TOOL_DESCRIPTION,
    API_CONTROLLER_TOOL_NAME,
    API_ORCHESTRATOR_PROMPT,
    API_PLANNER_PROMPT,
    API_PLANNER_TOOL_DESCRIPTION,
    API_PLANNER_TOOL_NAME,
    PARSING_GET_PROMPT,
    PARSING_POST_PROMPT,
    REQUESTS_GET_TOOL_DESCRIPTION,
    REQUESTS_POST_TOOL_DESCRIPTION,
)
from langchain.agents.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.requests import RequestsWrapper
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.tools.requests.tool import BaseRequestsTool

#
# Requests tools with LLM-instructed extraction of truncated responses.
#
# Of course, truncating so bluntly may lose a lot of valuable
# information in the response.
# However, the goal for now is to have only a single inference step.
MAX_RESPONSE_LENGTH = 5000


class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests_get"
    description = REQUESTS_GET_TOOL_DESCRIPTION
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain = LLMChain(
        llm=OpenAI(),
        prompt=PARSING_GET_PROMPT,
    )

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.get(data["url"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        return self._run(text)
        # raise NotImplementedError()
        # try:
        #     data = json.loads(text)
        # except json.JSONDecodeError as e:
        #     raise e
        # response = await self.requests_wrapper.aget(data["url"])
        # response = response[: self.response_length]
        # return await self.llm_chain.apredict(
        #     response=response, instructions=data["output_instructions"]
        # )


class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests_post"
    description = REQUESTS_POST_TOOL_DESCRIPTION

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain = LLMChain(
        llm=OpenAI(),
        prompt=PARSING_POST_PROMPT,
    )

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        return self._run(text)
        # raise NotImplementedError()
        # try:
        #     data = json.loads(text)
        # except json.JSONDecodeError as e:
        #     raise e
        # response = await self.requests_wrapper.apost(data["url"], data["data"])
        # response = response[: self.response_length]
        # return await self.llm_chain.apredict(
        #     response=response, instructions=data["output_instructions"]
        # )


#
# Orchestrator, planner, controller.
#
def _create_api_planner_tool(
        api_spec: ReducedOpenAPISpec, llm: BaseLanguageModel, plugins: Optional[dict]
) -> Tool:
    endpoint_descriptions = [
        f"{name} {description}" for name, description, _ in api_spec.endpoints
    ]
    prompt = PromptTemplate(
        template=API_PLANNER_PROMPT,
        input_variables=["query"],
        partial_variables={"endpoints": "- " + "- ".join(endpoint_descriptions)},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tool = Tool(
        name=plugins["name"] + " " + API_PLANNER_TOOL_NAME,
        description=API_PLANNER_TOOL_DESCRIPTION.format(plugins["name"], plugins["description"]),
        coroutine=chain.arun,
        func=chain.run
    )
    return tool


def _create_api_controller_agent(
        api_url: str,
        api_docs: str,
        requests_wrapper: RequestsWrapper,
        llm: BaseLanguageModel,
) -> AgentExecutor:
    tools: List[BaseTool] = [
        RequestsGetToolWithParsing(requests_wrapper=requests_wrapper),
        RequestsPostToolWithParsing(requests_wrapper=requests_wrapper),
    ]
    prompt = PromptTemplate(
        template=API_CONTROLLER_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "api_url": api_url,
            "api_docs": api_docs,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def _create_api_controller_tool(
        api_spec: ReducedOpenAPISpec,
        requests_wrapper: RequestsWrapper,
        llm: BaseLanguageModel,
        plugin: Optional[dict]
) -> Tool:
    """Expose controller as a tool.

    The tool is invoked with a plan from the planner, and dynamically
    creates a controller agent with relevant documentation only to
    constrain the context.
    """

    base_url = api_spec.servers[0]["url"]  # TODO: do better.

    def check_name_matches_names(names: dict, name: str) -> tuple:
        for endpoint_key in names.keys():
            method, path = endpoint_key.split(" ", 1)
            name_method, name_path = name.split(" ", 1)

            if method != name_method:
                continue

            endpoint_parts = path.strip("/").split("/")
            name_parts = name_path.strip("/").split("/")

            if len(endpoint_parts) != len(name_parts):
                continue

            match = True

            for endpoint_part, name_part in zip(endpoint_parts, name_parts):
                if endpoint_part.startswith("{") and endpoint_part.endswith("}"):
                    param_info = names[endpoint_key]
                    param_name = endpoint_part[1:-1]

                    param = None
                    for p in param_info["parameters"]:
                        if p["name"] == param_name:
                            param = p

                    if param and "enum" in param.get("schema", {}):
                        if name_part not in param["schema"]["enum"]:
                            match = False
                            break
                elif endpoint_part != name_part:
                    match = False
                    break

            if match:
                return True, endpoint_key

        return False, None

    def _create_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            matching, matched_key = check_name_matches_names(endpoint_docs_by_name, endpoint_name)
            if not matching:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")

            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(endpoint_docs_by_name.get(matched_key))}\n"
        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm)
        return agent.run(plan_str)

    async def _acreate_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            matching, matched_key = check_name_matches_names(endpoint_docs_by_name, endpoint_name)
            if not matching:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")

            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(endpoint_docs_by_name.get(matched_key))}\n"
        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm)
        return await agent.arun(plan_str)

    return Tool(
        name=plugin["name"] + " " + API_CONTROLLER_TOOL_NAME,
        coroutine=_acreate_and_run_api_controller_agent,
        description=API_CONTROLLER_TOOL_DESCRIPTION.format(plugin["name"]),
        func=_create_and_run_api_controller_agent
    )


def create_openapi_agent(
        api_spec: ReducedOpenAPISpec,
        requests_wrapper: RequestsWrapper,
        llm: BaseLanguageModel,
) -> AgentExecutor:
    """Instantiate API planner and controller for a given spec.

    Inject credentials via requests_wrapper.

    We use a top-level "orchestrator" agent to invoke the planner and controller,
    rather than a top-level planner
    that invokes a controller with its plan. This is to keep the planner simple.
    """
    tools = [
        _create_api_planner_tool(api_spec, llm),
        _create_api_controller_tool(api_spec, requests_wrapper, llm),
    ]
    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def create_openapi_agent_by_list(
        api_specs: List[ReducedOpenAPISpec],
        requests_wrapper: RequestsWrapper,
        llm: BaseLanguageModel,
        plugins: List[dict]
) -> AgentExecutor:
    """Instantiate API planner and controller for a given spec.

    Inject credentials via requests_wrapper.

    We use a top-level "orchestrator" agent to invoke the planner and controller,
    rather than a top-level planner
    that invokes a controller with its plan. This is to keep the planner simple.
    """
    tools = []
    for index, api_spec in enumerate(api_specs):
        tools.append(_create_api_planner_tool(api_spec, llm, plugins[index]))
        tools.append(_create_api_controller_tool(api_spec, requests_wrapper, llm, plugins[index]))

    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
