from pathlib import Path
import yaml
from dotenv import load_dotenv
from core.schemas import ModelSpec

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_registry(path: Path | None = None) -> tuple[list[ModelSpec], float]:
    data = yaml.safe_load((path or ROOT / "config" / "model_registry.yaml").read_text())
    models = [ModelSpec(**m) for m in data["models"]]   # pydantic validates here
    return models, float(data.get("local_cost_per_mtok_estimate", 0.0))


def load_routing_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config" / "routing_config.yaml").read_text())
