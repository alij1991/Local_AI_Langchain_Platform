from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

import httpx

from .config import AppConfig


@dataclass
class CommandResult:
    ok: bool
    output: str


class LMStudioController:
    """Control LM Studio via CLI commands and HTTP API endpoints."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _run_cli(self, command: str) -> CommandResult:
        full_command = [self.config.lm_studio_cli_bin, *shlex.split(command)]
        try:
            proc = subprocess.run(full_command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return CommandResult(False, f"CLI not found: {self.config.lm_studio_cli_bin}")

        output = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if err:
            output = f"{output}\n{err}".strip()
        return CommandResult(proc.returncode == 0, output)

    def start_server(self) -> CommandResult:
        return self._run_cli(self.config.lm_studio_cli_server_start)

    def stop_server(self) -> CommandResult:
        return self._run_cli(self.config.lm_studio_cli_server_stop)

    def load_model(self, model_name: str) -> CommandResult:
        cmd = self.config.lm_studio_cli_model_load_template.format(model=model_name)
        return self._run_cli(cmd)

    def list_local_models(self) -> CommandResult:
        return self._run_cli(self.config.lm_studio_cli_list_models)

    def list_loaded_models(self) -> CommandResult:
        url = f"{self.config.lm_studio_base_url.rstrip('/')}/models"
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url, headers={"Authorization": f"Bearer {self.config.lm_studio_api_key}"})
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            models = [item.get("id", "") for item in payload.get("data", []) if item.get("id")]
            return CommandResult(True, "\n".join(models) if models else "No loaded models found.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    @staticmethod
    def parse_model_lines(output: str) -> list[str]:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return []

        # Try JSON first for CLIs that support --json output.
        try:
            payload = json.loads(output)
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                return [str(item.get("id")) for item in payload["data"] if item.get("id")]
            if isinstance(payload, list):
                return [str(item.get("id", item)) if isinstance(item, dict) else str(item) for item in payload]
        except json.JSONDecodeError:
            pass

        return lines
