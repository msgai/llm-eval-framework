"""
eval_utils.py — Utility helpers for llm-eval-framework.

All URLs, credentials, and tuneable parameters are read from config.yaml;
nothing is hardcoded here.
"""

import json
import os
from typing import Any, Dict, List, Optional
import asyncio
import requests
import yaml
from config_proxy.data_manager.dm_proxy import DMProxy

# Resolved at import time — always points to this file's directory.
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


# Configuration helpers

def load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    """Resolve file path."""
    if os.path.isabs(path):
        return path
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, path)


# DMProxy factory

def fetch_config_map(env: str = "qa") -> dict:
    """Fetch live configuration map from config service."""
    cfg = load_config()
    svc = cfg["config_service"]

    env_key = env.lower() if env.lower() in ("local", "qa", "dev", "prod") else "qa"
    url = svc[env_key]["dm_config_url"]
    env_val = svc[env_key]["env"]
    service_name = svc["service_name"]
    region = svc["region"]
    timeout = int(svc.get("timeout_seconds", 10))

    payload = {
        "serviceName": service_name,
        "env": env_val,
        "region": region,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("payload", {}).get("configMap", {})
        print(f"[fetch_config_map] HTTP {response.status_code} from config service ({env})")
    except Exception as exc:
        print(f"[fetch_config_map] Error fetching config for env '{env}': {exc}")
    return {}


def get_dm_config_proxy(env: str = "qa") -> DMProxy:
    """Build DMProxy instance for the environment."""
    cfg = load_config()
    defs = cfg["dm_proxy_defaults"]
    env_defs = defs.get(env.lower() if env.lower() in ("qa", "dev") else "dev", {})

    config_map = fetch_config_map(env)

    redis_host = config_map.get("REDIS_HOST", env_defs.get("redis_host", ""))
    redis_port = int(config_map.get("REDIS_PORT", defs.get("redis_port", 6379)))
    redis_password = config_map.get("REDIS_PASSWORD", defs.get("redis_password", ""))
    datamanager_url = config_map.get(
        "DATA_MANAGER_ENDPOINT", env_defs.get("datamanager_url", "")
    )
    client_secret_key = config_map.get(
        "DATA_MANAGER_CLIENT_SECRET", defs.get("client_secret_key", "")
    )

    cache_ttl = int(config_map.get("DATA_MANAGER_CACHE_TTL", defs.get("cache_ttl", 300)))
    api_timeout = int(
        config_map.get(
            "DATA_MANAGER_API_TIMEOUT",
            config_map.get(
                "DATA_MANAGER_APIS_TIMEOUT_SECONDS", defs.get("api_timeout", 5)
            ),
        )
    )
    total_retries = int(
        config_map.get("DATA_MANAGER_APIS_TOTAL_RETRIES", defs.get("total_retries", 3))
    )

    max_concurrent_tasks = int(
        config_map.get("DM_PROXY_MAX_CONCURRENCY", defs.get("max_concurrency", 3))
    )
    local_cache_maxsize = int(
        config_map.get("DM_PROXY_IN_MEMORY_CACHE_SIZE", defs.get("in_memory_cache_size", 5000))
    )
    local_cache_ttl = int(
        config_map.get("DM_PROXY_CACHE_TTL", defs.get("in_memory_cache_ttl", 3600))
    )

    use_cache_flag_raw = config_map.get(
        "DM_PROXY_USE_CACHE_FLAG", str(defs.get("use_in_memory_cache", False))
    )
    use_in_memory_cache = str(use_cache_flag_raw).lower() == "true"

    max_in_memory_cache_entry_size = int(
        config_map.get(
            "DM_PROXY_MAX_IN_MEMORY_CACHE_ENTRY_SIZE",
            defs.get("max_in_memory_cache_entry_size", 102400),
        )
    )

    return DMProxy(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        datamanager_url=datamanager_url,
        client_secret_key=client_secret_key,
        cache_ttl=cache_ttl,
        api_timeout=api_timeout,
        total_retries=total_retries,
        max_concurrent_tasks=max_concurrent_tasks,
        local_cache_maxsize=local_cache_maxsize,
        local_cache_ttl=local_cache_ttl,
        use_in_memory_cache=use_in_memory_cache,
        max_in_memory_cache_entry_size=max_in_memory_cache_entry_size,
    )


# ---------------------------------------------------------------------------
# SOP / tool config loaders
# ---------------------------------------------------------------------------

def generate_sop_markdown(input_data: list, tool_map: dict) -> str:
    """
    Convert raw task-detail records from the DM API into the sops.txt
    markdown format expected by the evaluation prompt.

    action_id UUIDs in TOOLS_REQUIRED and ACTION_FLOW are resolved to
    human-readable [[Tool_name]] tokens via *tool_map*.
    Unknown action_ids are left as-is (graceful degradation).
    """
    markdown_output = []

    for index, task in enumerate(input_data, start=1):
        # 1. SOP title
        task_label = task.get("taskLabel", "Untitled Task")
        title = f"### **SOP {index}: {task_label}**"

        # 2. Trigger condition — normalise to a bulleted list
        trigger_condition = task.get("triggerCondition", "None provided.").strip()
        trigger_lines = []
        for line in trigger_condition.split("\n"):
            line_str = line.strip()
            if line_str:
                if not line_str.startswith("-") and not line_str.startswith("*"):
                    trigger_lines.append(f"- {line_str}")
                else:
                    trigger_lines.append(line_str)
        trigger = "#### **TRIGGER_CONDITION**:\n" + "\n".join(trigger_lines)

        # 3. Tools required — deduplicate, resolve action_ids → [[Name]]
        tools_raw = task.get("tools", "")
        tools_formatted = "#### **TOOLS_REQUIRED**:\n"
        if tools_raw and tools_raw.strip():
            tools_list = [t.strip() for t in tools_raw.split(",")]
            # Deduplicate preserving order
            seen: set = set()
            unique_tools = []
            for t in tools_list:
                if t not in seen:
                    seen.add(t)
                    unique_tools.append(t)

            for tool in unique_tools:
                resolved_name = tool_map.get(tool, tool)
                if resolved_name.startswith("[[") and resolved_name.endswith("]]"):
                    tools_formatted += f"    - {resolved_name}\n"
                elif "-" in resolved_name and len(resolved_name) > 20:
                    # Likely an unresolved UUID — print as-is
                    tools_formatted += f"    - {resolved_name}\n"
                else:
                    tools_formatted += f"    - [[{resolved_name}]]\n"
        else:
            tools_formatted += "    - None\n"

        # 4. Action flow — replace action_id occurrences with [[Name]]
        task_flow = task.get("taskFlow", "No action flow provided.").replace("\xa0", " ")
        for action_id, name in tool_map.items():
            if action_id in task_flow:
                task_flow = task_flow.replace(f"[[{action_id}]]", f"[[{name}]]")
                task_flow = task_flow.replace(action_id, f"[[{name}]]")

        action_flow = f"#### **ACTION_FLOW**:\n{task_flow}"

        sop_section = f"{title}\n\n{trigger}\n\n{tools_formatted}\n{action_flow}"
        markdown_output.append(sop_section)

    return "\n\n".join(markdown_output) + "\n\n\n---\n"


def load_tool_configs(
    bot_id: str,
    bot_ref_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return the list of tool definitions for the given bot.

    Tries to retrieve them dynamically from DMProxy (using the
    bot_env and is_action_v2_enabled settings from config.yaml).
    """
    cfg = load_config()
    env = cfg.get("bot", {}).get("env") or os.environ.get("ENV_CONFIG", "qa")
    dm_proxy = get_dm_config_proxy(env)
    api_cfg = cfg.get("dm_proxy_api", {})
    bot_env = api_cfg.get("bot_env", "SANDBOX")
    is_action_v2 = bool(api_cfg.get("is_action_v2_enabled", True))

    tools = asyncio.run(
        dm_proxy.get_tools_config(
            bot_id=bot_id,
            bot_ref_id=bot_ref_id,
            env=bot_env,
            is_action_v2_enabled=is_action_v2,
            use_cache=False,
        )
    )
    if tools:
        return tools
    raise RuntimeError("No tools fetched dynamically.")


def load_sop_configs(
    bot_id: str,
    bot_ref_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return the AVAILABLE_SOP text block for the given bot.

    Retrieves and formats dynamically from DMProxy.
    """
    cfg = load_config()
    env = cfg.get("bot", {}).get("env") or os.environ.get("ENV_CONFIG", "qa")
    dm_proxy = get_dm_config_proxy(env)
    api_cfg = cfg.get("dm_proxy_api", {})
    bot_env = api_cfg.get("bot_env", "SANDBOX")
    is_action_v2 = bool(api_cfg.get("is_action_v2_enabled", True))

    # Build action_id → tool_name map
    tools = asyncio.run(
        dm_proxy.get_tools_config(
            bot_id=bot_id,
            bot_ref_id=bot_ref_id,
            env=bot_env,
            is_action_v2_enabled=is_action_v2,
            use_cache=False,
        )
    )
    tool_map: Dict[str, str] = {}
    if tools:
        for t in tools:
            action_id = t.get("action_id")
            t_name = t.get("tool_name")
            if action_id and t_name:
                formatted_name = t_name[0].upper() + t_name[1:] if t_name else t_name
                tool_map[action_id] = formatted_name

    # Fetch SOPs
    sops = asyncio.run(
        dm_proxy.get_task_details_config(
            bot_id=bot_id,
            bot_ref_id=bot_ref_id,
            env=bot_env,
            use_cache=False,
        )
    )
    if sops:
        return generate_sop_markdown(sops, tool_map)
    raise RuntimeError("No SOPs fetched dynamically.")


# Static file loaders

def load_eval_prompt(config: Optional[Dict[str, Any]] = None) -> str:
    """Return raw evaluation prompt template."""
    cfg = load_config()
    prompt_path = _resolve_path(cfg["prompts"]["eval_prompt_path"])
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# Dynamic prompt loaders via DMProxy

def format_adjust_response_rules(rules: list) -> str:
    """Convert adjustResponse rules to evaluation prompt format."""
    topics_list = ["'" + rule.get("title", "") + "'" for rule in rules]
    topics_str = " or ".join(topics_list)

    parts = []
    header = (
        "### *Adjust Response Guidelines**:\n"
        f"For the below given topics, TAILOR your responses as per the provided guidelines and **sample example response**. "
        f"First determine if the topic of the user message is explicitly about {topics_str}. "
        "If so, follow the below instructions VERY STRICTLY.\n \n    \n    "
    )
    parts.append(header)

    for rule in rules:
        title = rule.get("title", "")
        instruction = rule.get("instruction", "")
        example_response = rule.get("exampleResponse", "")

        rule_str = f"Topic: **{title}**\n    \n**Guideline:** '{instruction}'"
        if example_response:
            rule_str += f"\n            \n**Example Response:** '{example_response}'"

        parts.append(rule_str)

    return "\n    \n    \n".join(parts)

async def load_dynamic_instructions(
    bot_id: str,
    bot_ref_id: str,
    env: str = "qa",
) -> tuple[str, str]:
    """
    Fetch CUSTOM_TONE_INSTRUCTIONS and ADJUST_RESPONSE_RULES
    for the given bot.
    """

    cfg = load_config()

    # DM API configuration
    api_cfg = cfg.get("dm_proxy_api", {})
    capability_id = int(api_cfg.get("capability_id", 2))

    bot_env = api_cfg.get("bot_env", "SANDBOX")

    dm_proxy = get_dm_config_proxy(env)

    # Use fetch_prompts_list if available; fallback to 'le' due to a bug/rename in some library versions
    fetch_method = getattr(dm_proxy, "fetch_prompts_list", None) or getattr(dm_proxy, "le", None)
    if not fetch_method:
        raise AttributeError("Neither 'fetch_prompts_list' nor 'le' method found on dm_proxy")

    resp = await fetch_method(
        bot_id=bot_id,
        capability_id=capability_id,
        bot_ref_id=bot_ref_id,
        env=bot_env,
        use_cache=False,
    )

    if not resp:
        return "", ""

    prompt = next(
        (p for p in resp if p.get("capabilityTask") == "RESPONSE_GENERATOR"),
        resp[0],
    )

    basic_config = prompt.get("basicConfiguration", {})
    custom_tone_prompt = basic_config.get("customTonePrompt", "")

    advanced_config = prompt.get("advancedConfiguration", {})
    adjust_response_rules = advanced_config.get("adjustResponse", [])
    adjust_response_rules_str = format_adjust_response_rules(
        adjust_response_rules
    )

    return custom_tone_prompt, adjust_response_rules_str