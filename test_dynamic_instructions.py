import asyncio
import logging
import sys

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

# Configure logging to see all debug messages
logging.basicConfig(level=logging.DEBUG)

from eval_utils import load_config, load_dynamic_instructions

async def main():
    # Load config from config.yaml
    config = load_config()
    bot_id = config["bot"]["bot_id"]
    bot_ref_id = config["bot"]["bot_ref_id"]
    import os
    env = config["bot"].get("env") or os.environ.get("ENV_CONFIG") or "qa"
    
    print(f"Loaded bot_id: {bot_id}")
    print(f"Loaded bot_ref_id: {bot_ref_id}")
    print(f"Loaded env: {env}")
    print("Calling load_dynamic_instructions...")
    try:
        tone, rules = await load_dynamic_instructions(
            env=env,
            bot_id=bot_id,
            bot_ref_id=bot_ref_id
        )
        print("Tone length:", len(tone))
        print("Rules length:", len(rules))

        import os
        output_dir = "/Users/pankajkumar/Desktop/Netomi/llm-eval-framework/datasets"
        os.makedirs(output_dir, exist_ok=True)

        tone_path = os.path.join(output_dir, "custom_tone_instructions.txt")
        rules_path = os.path.join(output_dir, "adjust_response_rules.txt")

        with open(tone_path, "w", encoding="utf-8") as f:
            f.write(tone)
        with open(rules_path, "w", encoding="utf-8") as f:
            f.write(rules)

        print(f"Saved custom_tone_instructions to {tone_path}")
        print(f"Saved adjust_response_rules to {rules_path}")
    except Exception as e:
        print("Error caught in main:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
