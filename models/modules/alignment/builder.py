"""
Alignment plugin builder.

Current supported built-in module:
    identity

Future modules can be added WITHOUT changing the detector.

Two future usage modes
----------------------

A) Add a built-in name to BUILTIN_ALIGNMENTS.

B) No builder modification at all:
   point YAML `target` to a Python class:

alignment:
  enabled: true
  type: custom
  target: models.modules.alignment.offset_alignment:OffsetAlignment

The target class only needs to inherit BaseAlignment and obey its forward API.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Type

from .base import BaseAlignment
from .identity import IdentityAlignment


BUILTIN_ALIGNMENTS = {
    "identity":
        IdentityAlignment,
}


def _load_target(
    target: str,
) -> Type[BaseAlignment]:
    """
    Dynamic class loader.

    target format:
        package.module:ClassName
    """

    if ":" not in target:
        raise ValueError(
            "alignment.target 必须使用 "
            "'package.module:ClassName' 格式，"
            f"当前为: {target}"
        )

    module_name, class_name = (
        target.split(
            ":",
            1,
        )
    )

    module = importlib.import_module(
        module_name
    )

    if not hasattr(
        module,
        class_name,
    ):
        raise AttributeError(
            f"{module_name} 中不存在 "
            f"{class_name}"
        )

    cls = getattr(
        module,
        class_name,
    )

    if not isinstance(
        cls,
        type,
    ):
        raise TypeError(
            f"{target} 不是 Python class。"
        )

    if not issubclass(
        cls,
        BaseAlignment,
    ):
        raise TypeError(
            f"{target} 必须继承 BaseAlignment。"
        )

    return cls


def build_alignment(
    cfg: Dict[str, Any] | None = None,
) -> BaseAlignment:
    """
    Build an optional RGB-T alignment plugin.

    Important
    ---------
    enabled=false always returns IdentityAlignment.
    Therefore alignment can be disabled without any detector-side branch.
    """

    cfg = dict(
        cfg or {}
    )

    enabled = bool(
        cfg.get(
            "enabled",
            False,
        )
    )

    # -------------------------------------------------------
    # Current default: alignment OFF
    # -------------------------------------------------------
    if not enabled:
        return IdentityAlignment(
            cfg=cfg
        )

    align_type = str(
        cfg.get(
            "type",
            "identity",
        )
    ).strip().lower()

    # -------------------------------------------------------
    # Built-in plugins
    # -------------------------------------------------------
    if align_type in BUILTIN_ALIGNMENTS:
        cls = BUILTIN_ALIGNMENTS[
            align_type
        ]

        return cls(
            cfg=cfg
        )

    # -------------------------------------------------------
    # Future external/custom plugin
    # -------------------------------------------------------
    target = cfg.get(
        "target",
        None,
    )

    if not target:
        raise NotImplementedError(
            "\n当前尚未实现真正的对齐算法。\n"
            f"alignment.type={align_type!r}\n"
            "如果以后新增插件，请设置例如：\n"
            "alignment:\n"
            "  enabled: true\n"
            "  type: custom\n"
            "  target: "
            "models.modules.alignment."
            "offset_alignment:OffsetAlignment\n"
        )

    cls = _load_target(
        str(
            target
        )
    )

    return cls(
        cfg=cfg
    )
