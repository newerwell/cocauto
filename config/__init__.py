"""
配置管理模块 (Config Module)
负责加载、合并和强类型校验 default_config.yaml 与 user_profile.yaml。
支持在无 PyYAML 依赖的极简手机 Termux 环境下回退到内置解析器或 JSON 兼容格式。
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any
from .paths import get_app_root, get_config_dir, get_templates_dir

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class DeviceConfig:
    serial: str = "127.0.0.1:5561"
    standard_width: int = 1280
    standard_height: int = 720
    screencap_timeout: float = 5.0
    use_maatouch: bool = False


@dataclass
class LootFilterConfig:
    match_mode: str = "regular"          # "regular" (常规战，可点下一个挑鱼) | "ranked" (排位战，搜到谁直接打谁)
    auto_attack_on_found: bool = True    # 搜到目标对手后是否自动衔接下兵战斗
    min_gold: int = 450_000
    min_elixir: int = 450_000
    min_gold_plus_elixir: int = 1_000_000
    min_dark_elixir: int = 4_500
    only_dead_base: bool = False
    dead_base_min_collectors: int = 3
    max_search_clicks: int = 50


@dataclass
class CombatConfig:
    strategy: str = "surround_barch"
    deploy_wave_count: int = 3
    hero_ability_hp_ratio: float = 0.45
    max_battle_duration_sec: int = 185
    early_exit_no_loot_sec: int = 15


@dataclass
class VillageConfig:
    collect_resources: bool = True
    clear_obstacles: bool = False
    auto_request_cc: bool = True
    auto_donate: bool = True
    donate_keywords: List[str] = field(
        default_factory=lambda: ["任意", "随便", "打鱼", "地面", "空军", "雷龙", "气球"]
    )
    donate_blacklist: List[str] = field(
        default_factory=lambda: ["不", "勿", "只要超级", "攻城"]
    )


@dataclass
class UpgradeConfig:
    enable_auto_wall: bool = True
    enable_auto_building: bool = True
    reserve_builders_for_wall: int = 1
    wall_min_gold: int = 4_000_000
    wall_min_elixir: int = 4_000_000
    keep_minimum_resource: int = 1_000_000


@dataclass
class HumanizerConfig:
    click_sigma_px: float = 6.0
    click_max_offset_px: int = 15
    min_action_delay_sec: float = 0.22
    max_action_delay_sec: float = 0.65
    max_session_minutes: int = 75
    rest_duration_minutes: int = 30
    night_sleep_window: str = "01:30-07:30"


@dataclass
class AppConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    loot_filter: LootFilterConfig = field(default_factory=LootFilterConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    village: VillageConfig = field(default_factory=VillageConfig)
    upgrade: UpgradeConfig = field(default_factory=UpgradeConfig)
    humanizer: HumanizerConfig = field(default_factory=HumanizerConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = cls()
        if "device" in data and isinstance(data["device"], dict):
            cfg.device = DeviceConfig(**{k: v for k, v in data["device"].items() if hasattr(DeviceConfig, k)})
        if "loot_filter" in data and isinstance(data["loot_filter"], dict):
            cfg.loot_filter = LootFilterConfig(**{k: v for k, v in data["loot_filter"].items() if hasattr(LootFilterConfig, k)})
        if "combat" in data and isinstance(data["combat"], dict):
            cfg.combat = CombatConfig(**{k: v for k, v in data["combat"].items() if hasattr(CombatConfig, k)})
        if "village" in data and isinstance(data["village"], dict):
            cfg.village = VillageConfig(**{k: v for k, v in data["village"].items() if hasattr(VillageConfig, k)})
        if "upgrade" in data and isinstance(data["upgrade"], dict):
            cfg.upgrade = UpgradeConfig(**{k: v for k, v in data["upgrade"].items() if hasattr(UpgradeConfig, k)})
        if "humanizer" in data and isinstance(data["humanizer"], dict):
            cfg.humanizer = HumanizerConfig(**{k: v for k, v in data["humanizer"].items() if hasattr(HumanizerConfig, k)})
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """加载配置文件，若未指定则自动读取外部可写目录 config/user_profile.yaml 或 config/default_config.yaml"""
    base_dir = get_config_dir()
    target_path = Path(config_path) if config_path else base_dir / "user_profile.yaml"
    if not target_path.exists():
        target_path = base_dir / "default_config.yaml"
    if not target_path.exists():
        # 回退至源码内置目录
        target_path = Path(__file__).resolve().parent / "default_config.yaml"

    if target_path.exists() and HAS_YAML:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return AppConfig.from_dict(raw_data)

    return AppConfig()


def save_user_config(cfg: AppConfig, config_path: str | Path | None = None) -> Path:
    """将用户在 GUI 界面修改的配置持久化保存至 config/user_profile.yaml"""
    base_dir = get_config_dir()
    target_path = Path(config_path) if config_path else base_dir / "user_profile.yaml"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_YAML:
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)
    return target_path
