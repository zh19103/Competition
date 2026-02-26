import os

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import dashscope
except Exception:
    dashscope = None

from .config import QwenConfig

class QwenGenerator:
    def __init__(self, cfg: QwenConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        self.client = None
        self.model = None
        if self.cfg.use_openai:
            if OpenAI is None:
                raise RuntimeError("缺少 openai 库，请 pip install openai>=1.0.0")
            api_key = (self.cfg.openai_api_key or
           	       os.environ.get(self.cfg.openai_api_key_env, ""))
            if not api_key:
                raise RuntimeError(f"缺少环境变量 {self.cfg.openai_api_key_env}或QwenConfig.openai_api_key")
            self.client = OpenAI(base_url=self.cfg.openai_base_url, api_key=api_key)
            self.model = self.cfg.openai_model
        else:
            if dashscope is None:
                raise RuntimeError("缺少 dashscope 库，请 pip install dashscope")
            api_key = os.environ.get(self.cfg.dashscope_api_key_env, "")
            if not api_key:
                raise RuntimeError(f"缺少环境变量 {self.cfg.dashscope_api_key_env}")
            dashscope.api_key = api_key
            self.model = self.cfg.dashscope_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.client is not None:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        else:
            result = dashscope.Generation.call(
                model=self.model,
                input={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                },
                parameters={
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "max_tokens": self.cfg.max_tokens,
                }
            )
            if result.status_code == 200:
                return result.output["text"].strip()
            raise RuntimeError(f"DashScope 调用失败: {result.code}, {result.message}")
