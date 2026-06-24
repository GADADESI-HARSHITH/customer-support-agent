# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any
import google.auth
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, Edge, START, node
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

# Clean up API key if it contains literal double quotes (common shell wrapper issue)
if "GEMINI_API_KEY" in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"].strip('"')
    # If using developer API, default to AI Studio (not Vertex AI)
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    # Set up GCP project and location for Vertex AI
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    except Exception:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "mock-project")

    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


class Classification(BaseModel):
    category: str = Field(
        description="Must be 'shipping' if the query is related to tracking, delivery, shipping rates, or returns. Otherwise, must be 'general'."
    )
    reason: str = Field(description="Explanation for the classification.")


# Disable thinking mode to minimize latency and prevent test timeouts
no_thinking_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)

# Specialized classifier agent
classifier_agent = LlmAgent(
    name="classifier_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert customer request classifier for a shipping company.\n"
        "Your task is to classify user queries into one of two categories:\n"
        "1. 'shipping': If the query is related to shipping topics such as tracking packages, delivery status/times, shipping rates/costs, or returns/refunds.\n"
        "2. 'general': For everything else, including general questions, greetings, feedback, or general conversation.\n\n"
        "Analyze the user query carefully and choose the correct category."
    ),
    output_schema=Classification,
    generate_content_config=no_thinking_config,
)

# Specialized Shipping FAQ agent
shipping_faq_agent = LlmAgent(
    name="shipping_faq_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a super friendly and helpful customer support representative for a shipping company, specializing in shipping FAQs! 😊\n"
        "Answer shipping-related questions warmly and professionally, using friendly emojis (like 📦, 🚚, 🕒, 💸, 🔄) to keep it engaging. ✨\n"
        "Topics include:\n"
        "- tracking: tell users how to track packages (e.g. check their tracking link, call our helpline, etc.) 📦\n"
        "- delivery: answer questions about delivery times, missing packages, scheduling deliveries 🚚\n"
        "- shipping rates: explain costs, calculations, weight/size limits. Always mention our free shipping threshold for orders over $50! 💸\n"
        "- returns: details on how to return items, print labels, return policies 🔄\n\n"
        "If you don't have enough details (like a tracking number), ask the user for them politely and warmly. 💕\n"
        "Keep your response polite, professional, warm, and concise."
    ),
    generate_content_config=no_thinking_config,
)

# General Assistant agent
general_assistant = LlmAgent(
    name="general_assistant",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a helpful general assistant for a shipping company customer support desk.\n"
        "You handle general inquiries, greetings, or questions unrelated to shipping.\n"
        "Respond politely and professionally, guiding the user to shipping-related help if they eventually ask for it.\n"
        "Keep your response polite, professional, and concise."
    ),
    generate_content_config=no_thinking_config,
)


def get_user_text(node_input: Any) -> str:
    if isinstance(node_input, str):
        return node_input
    if hasattr(node_input, "text"):
        return node_input.text
    if hasattr(node_input, "parts") and node_input.parts:
        return node_input.parts[0].text
    if isinstance(node_input, dict) and "text" in node_input:
        return node_input["text"]
    return str(node_input)


@node(rerun_on_resume=True)
async def classify_request(ctx: Context, node_input: Any) -> Event:
    # Run the classifier agent to get the classification
    result = await ctx.run_node(classifier_agent, node_input=node_input)
    category = result.get("category", "general")
    if category == "shipping":
        return Event(output=node_input, actions=EventActions(route="shipping"))
    return Event(output=node_input, actions=EventActions(route="general"))


# Workflow graph topology definition using explicit Edge objects
root_agent = Workflow(
    name="customer_support_workflow",
    edges=[
        Edge(from_node=START, to_node=classify_request),
        Edge(from_node=classify_request, to_node=shipping_faq_agent, route="shipping"),
        Edge(from_node=classify_request, to_node=general_assistant, route="general"),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
