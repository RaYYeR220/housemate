"""Housemate: hands-free Hermes in every room.

Kept import-light on purpose: Hermes defers ``kind: platform`` plugins, so the
adapter (and everything it pulls in: the LAN hub, audio pipeline, models) is
imported only when the gateway asks for the ``housemate`` platform. The agent
tools, slash commands, CLI, blueprints and skill register from ``tools.py``
(declared in ``provides_tools``) so they exist in every process.
"""

from __future__ import annotations

PLATFORM_NAME = "housemate"

PLATFORM_HINT = (
    "You are speaking out loud through a speaker in the user's home. Answer in one "
    "to three short spoken sentences, no markdown, no lists, no URLs read aloud. "
    "The room the user spoke from is given as [room: ...]. For long content "
    "(recipes, lists, steps) call housemate_show to put it on that room's screen "
    "and say one sentence about it."
)


def _adapter_factory(config):
    from .adapter import HousemateAdapter

    return HousemateAdapter(config)


def check_requirements() -> bool:
    """Passive probe: are the runtime dependencies importable right now?"""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import cryptography  # noqa: F401
    except Exception:
        return False
    return True


def register(ctx) -> None:
    # When Hermes defers the platform it already ran tools.register_tools for this
    # same load (same manifest object); never register the tools surface twice.
    try:
        from .tools import register_tools, registered_for

        if not registered_for(ctx):
            register_tools(ctx)
    except Exception:
        import logging

        logging.getLogger(__name__).warning("housemate: registering tools failed", exc_info=True)

    kwargs = dict(
        name=PLATFORM_NAME,
        label="Housemate",
        adapter_factory=_adapter_factory,
        check_fn=check_requirements,
        cron_deliver_env_var="HOUSEMATE_HOME_CHANNEL",
        allowed_users_env="HOUSEMATE_ALLOWED_DEVICES",
        allow_all_env="HOUSEMATE_ALLOW_ALL_DEVICES",
        max_message_length=0,
        platform_hint=PLATFORM_HINT,
        emoji="\U0001F3E0",
        install_hint="Enable the housemate plugin so Hermes installs its Python dependencies, then restart the gateway.",
    )
    try:
        import dataclasses

        from gateway.platform_registry import PlatformEntry

        known = {f.name for f in dataclasses.fields(PlatformEntry)}
        for key, value in (("display_tier", "minimal"),):
            if key in known:
                kwargs[key] = value
    except Exception:
        pass
    ctx.register_platform(**kwargs)
