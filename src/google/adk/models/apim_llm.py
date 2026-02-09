# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Azure APIM LLM implementation for ADK."""

from __future__ import annotations

from functools import cached_property
import os
from typing import Optional, TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .google_llm import Gemini

if TYPE_CHECKING:
  from google.genai import Client


class ApimLlm(Gemini):
  """A BaseLlm implementation for calling Azure APIM.

  Attributes:
    model: The name of the Gemini model.
  """

  def __init__(
      self,
      *,
      model: str,
      apim_url: str,
      apim_key: str,
      retry_options: Optional[types.HttpRetryOptions] = None,
  ):
    """Initializes the APIM LLM backend.

    Args:
      model: The model ID (e.g., 'gemini-2.5-flash').
      apim_url: The base URL of the Azure APIM endpoint.
      apim_key: The APIM subscription key.
      retry_options: Allow google-genai to retry failed responses.
    """
    super().__init__(model=model, retry_options=retry_options)
    self._apim_url = apim_url
    self._apim_key = apim_key
    self._api_version = 'v1beta'

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models."""
    return [r'gemini-.*', r'apim-.*']

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client."""
    from google.genai import Client

    http_options = types.HttpOptions(
        base_url=self._apim_url,
        headers={'Ocp-Apim-Subscription-Key': self._apim_key},
        api_version=self._api_version,
        retry_options=self.retry_options,
    )

    return Client(
        http_options=http_options,
        vertexai=False,
        api_key="dummy",  # Required but not used since we use custom headers
    )
