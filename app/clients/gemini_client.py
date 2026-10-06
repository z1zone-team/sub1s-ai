import asyncio
import logging
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
import json_repair

from app.configs.settings import settings
from app.prompts.briefing_prompt import (
    BRIEFING_SYSTEM_PROMPT,
    build_briefing_user_prompt,
)
from app.prompts.nlu_prompt import NLU_SYSTEM_PROMPT, build_nlu_user_prompt

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Google Gemini API 비동기 클라이언트 래퍼.
    Single-Call 구조, 2초 SLA 타임아웃, json-repair 응답 복구 및 장애 격리(Fault Tolerance)를 지원합니다.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self._client: Optional[genai.Client] = None
        if self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"[GeminiClient] Failed to initialize genai.Client: {e}")

    def _sync_generate_nlu(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None

        config = types.GenerateContentConfig(
            system_instruction=NLU_SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
        )

        response = self._client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )

        if not response or not response.text:
            return None

        parsed = json_repair.loads(response.text)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _sync_generate_briefing(self, prompt: str) -> Optional[str]:
        if not self._client:
            return None

        config = types.GenerateContentConfig(
            system_instruction=BRIEFING_SYSTEM_PROMPT,
            temperature=0.3,
        )

        response = self._client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )

        if not response or not response.text:
            return None

        return response.text.strip()

    async def generate_nlu(
        self, user_text: str, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Gemini 모델을 호출하여 NLU 슬롯 필링 JSON을 추출합니다.
        2초 타임아웃 또는 API 에러 시 None을 반환하여 파이프라인의 제로토큰 Fallback으로 연결합니다.
        """
        if not self._client or not self.api_key:
            return None

        timeout_sec = timeout or settings.nlu_timeout_sec
        user_prompt = build_nlu_user_prompt(user_text)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._sync_generate_nlu, user_prompt),
                timeout=timeout_sec,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"[GeminiClient] SLA timeout ({timeout_sec}s) exceeded for NLU request."
            )
            return None
        except Exception as e:
            logger.warning(f"[GeminiClient] Gemini API call failed: {e}")
            return None

    async def generate_briefing(
        self,
        origin: str,
        dest: str,
        mode_label: str,
        duration_min: int,
        transfer_count: int,
        tips: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """
        Gemini 모델을 호출하여 자연스러운 2문장 음성 브리핑 멘트를 생성합니다.
        2초 타임아웃 또는 에러 시 None을 반환하여 규칙 기반 템플릿으로 연결합니다.
        """
        if not self._client or not self.api_key:
            return None

        timeout_sec = timeout or settings.nlu_timeout_sec
        prompt = build_briefing_user_prompt(
            origin, dest, mode_label, duration_min, transfer_count, tips
        )

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._sync_generate_briefing, prompt),
                timeout=timeout_sec,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"[GeminiClient] SLA timeout ({timeout_sec}s) exceeded for Briefing request."
            )
            return None
        except Exception as e:
            logger.warning(f"[GeminiClient] Gemini Briefing call failed: {e}")
            return None
