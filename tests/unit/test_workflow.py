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

from google.genai import types

from app.agent import get_user_text


def test_get_user_text_str() -> None:
    assert get_user_text("hello") == "hello"


def test_get_user_text_content() -> None:
    content = types.Content(
        role="user", parts=[types.Part.from_text(text="hello content")]
    )
    assert get_user_text(content) == "hello content"


def test_get_user_text_dict() -> None:
    assert get_user_text({"text": "hello dict"}) == "hello dict"
