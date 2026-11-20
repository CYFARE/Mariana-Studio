"""The Studio application: every service, constructed once and shared by handlers."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ..agents.manager import AgentManager
from .chat_swarm import ChatSwarm
from .chats import ChatStore
from .dependencies import DependencyManager
from .deploy import DeployService
from .finetune import FinetuneService
from .ide import IdeService
from .jobs import JobRegistry
from .models import ModelGateway
from .models.downloads import DownloadManager
from .monitor import SystemMonitor
from .settings import SettingsStore
from .training import TrainingJobs, TrainingManager
from .updates import UpdateService
from .voice import VoiceService

NAME = "Mariana Studio"


class Studio:
    """Service container.  Handlers receive it as their first argument."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.port: int | None = None
        self.started = time.time()
        self.monitor = SystemMonitor(self.root, self.started)
        self._close_lock = threading.Lock()
        self._closed = False
        self.settings = SettingsStore(self.root)
        self.jobs = JobRegistry()
        self.training = TrainingManager(self.root, registry=self.jobs)
        self.jobs.add_provider(TrainingJobs(self.training))
        self.gateway = ModelGateway(self.root, settings=self.settings.get, own_port=lambda: self.port)
        self.gateway.training_active = lambda: self.training.active
        self.training.before_start.append(self.gateway.release_gpu)
        self.downloads = DownloadManager(
            self.root, self.jobs, on_complete=self.gateway.invalidate,
            ollama_url=self.gateway.ollama.url,
        )
        self.finetune = FinetuneService(self.root, self.gateway, self.training, self.jobs)
        self.agents = AgentManager(self.root, self.gateway, settings=self.settings.get)
        self.ide = IdeService(self.root, self.gateway, settings=self.settings.get)
        self.voice = VoiceService(self.root, self.jobs)
        self.chats = ChatStore(self.root)
        self.chat_swarm = ChatSwarm(self.root, self.gateway, settings=self.settings.get)
        self.deploy = DeployService(self.root, self.jobs, self.gateway)
        self.dependencies = DependencyManager(self.root, self.jobs, self.settings.get)
        self.updates = UpdateService(self.root, self.jobs, self.settings.get)

    def close(self) -> None:
        """Stop every thread and child process the services started, releasing GPU memory."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.monitor.close()
        for service in (self.ide, self.voice, self.agents, self.chat_swarm, self.training,
                        self.updates, self.jobs, self.gateway):
            try:
                service.close()
            except Exception:  # noqa: BLE001 - shutdown must reach every service
                pass
