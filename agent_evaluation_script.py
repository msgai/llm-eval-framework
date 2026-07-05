import asyncio
import argparse
import json
import logging  # noqa: F401  (kept for downstream use)
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pandarallel import pandarallel
from pydantic import BaseModel, Field

import eval_utils

# Bootstrap
load_dotenv()
os.environ["no_proxy"] = "*"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Parallel evaluation workers
pandarallel.initialize(progress_bar=True, nb_workers=8)

# Config loading
CONFIG = eval_utils.load_config()

BOT_ID = CONFIG["bot"]["bot_id"]
BOT_REF_ID = CONFIG["bot"]["bot_ref_id"]
LLM_NAME = CONFIG["llm_name"]
FILTER_TAGS_SOURCE_UUIDS = CONFIG["filter_tags_source_uuids"]

LOCAL_URL = CONFIG["api"]["local_url"]
HEADER = CONFIG["api"]["headers"]
REQUEST_TIMEOUT = CONFIG["api"]["request_timeout"]
SLEEP_TIMER = CONFIG["api"]["sleep_timer"]

EVAL_MODEL_NAME = CONFIG["evaluation"]["model_name"]
EVAL_REASONING_EFFORT = CONFIG["evaluation"]["reasoning_effort"]
EVAL_BATCH_SIZE = CONFIG["evaluation"]["batch_size"]
EVAL_DELAY_SECONDS = CONFIG["evaluation"]["delay_seconds"]

_DEFAULTS = CONFIG.get("defaults", {})
DEFAULT_MODEL_NAME = _DEFAULTS.get("model_name")
DEFAULT_DATASET_PATH = _DEFAULTS.get("dataset_path", "datasets/llm_eval_dataset_1.pkl")
DEFAULT_OUTPUT_DIR = _DEFAULTS.get("output_dir", "results")
DEFAULT_SKIP_GENERATION = _DEFAULTS.get("skip_generation", False)
DEFAULT_SKIP_EVALUATION = _DEFAULTS.get("skip_evaluation", False)
DEFAULT_RESULTS_FILE = _DEFAULTS.get("results_file")

# Dynamic data loading
TOOL_DATASET = eval_utils.load_tool_configs(BOT_ID, BOT_REF_ID, CONFIG)
AVAILABLE_SOP = eval_utils.load_sop_configs(BOT_ID, BOT_REF_ID, CONFIG)

# Load prompts dynamically from all tone_bots combinations in config.yaml
ENV_CONFIG = os.environ.get("ENV_CONFIG", "qa")
print(f"Dynamically loading prompts from '{ENV_CONFIG}' configuration for all tone_bots …")
CUSTOM_TONE_INSTRUCTIONS, ADJUST_RESPONSE_RULES = asyncio.run(
    eval_utils.load_dynamic_instructions_for_all_bots(env=ENV_CONFIG)
)
print("✔  Loaded and merged CUSTOM_TONE_INSTRUCTIONS and ADJUST_RESPONSE_RULES from DM API.")

EVALUATION_PROMPT = eval_utils.load_eval_prompt(CONFIG)

# Validate environment
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please set it in your .env file or as an environment variable.")
    raise SystemExit(1)


# Pydantic evaluation models

class ToolInvocationEvaluation(BaseModel):
    """Evaluation of individual tool invocation"""
    tool_name: str = Field(description="Name of the tool that was invoked")
    is_correct_tool: bool = Field(description="Whether the correct tool was chosen for the task")
    input_parameters_correct: bool = Field(description="Whether the input parameters were correct and complete")
    reasoning: str = Field(description="Brief explanation of why the tool invocation was correct or incorrect (1-2 sentences max)")


class ResponseQualityEvaluation(BaseModel):
    """Evaluation of the final response quality"""
    addresses_user_query: bool = Field(description="Whether the response adequately addresses the user's query")
    uses_tool_output_correctly: bool = Field(description="Whether the response correctly uses information from tool outputs")
    has_hallucination: bool = Field(description="Whether the response contains information not present in tool outputs")
    is_complete: bool = Field(description="Whether the response provides complete information requested by user")
    hallucination_details: Optional[str] = Field(description="Specific hallucinated content only (null if none)")


class SOPEvaluation(BaseModel):
    """Evaluation of SOP adherence"""
    sop_identified: Optional[str] = Field(description="The SOP that should have been triggered based on user query")
    sop_trigger_accuracy: float = Field(description="Score from 1-10 for correct SOP identification and triggering", ge=1, le=10)
    sop_execution_accuracy: float = Field(description="Score from 1-10 for following SOP steps in correct order with proper tool usage", ge=1, le=10)
    reasoning: str = Field(description="Concise explanation including: which SOP should be triggered, key violations if any, conversation history usage (2-3 sentences max)")


class CustomToneEvaluation(BaseModel):
    """Evaluation of custom tone adherence"""
    brand_voice_score: float = Field(description="Score from 1-10 for maintaining Equinox's elevated brand identity and avoiding forbidden phrases/terms", ge=1, le=10)
    terminology_score: float = Field(description="Score from 1-10 for using correct terminology (clubs/signature experiences/Members) and proper hyperlink formatting", ge=1, le=10)
    reasoning: str = Field(description="Concise explanation including: specific forbidden phrases or terms used, incorrect terminology, hyperlink formatting issues, and overall brand voice assessment (2-3 sentences max)")


class AdjustResponseEvaluation(BaseModel):
    """Evaluation of Adjust Response rule adherence"""
    rule_identified: Optional[str] = Field(description="The specific Adjust Response rule that should have been triggered based on user query")
    rule_trigger_accuracy: float = Field(description="Score from 1–10 for correctly identifying and applying the relevant Adjust Response rule", ge=1, le=10)
    rule_execution_accuracy: float = Field(description="Score from 1–10 for following the rule's instructions precisely (links, phrasing, restrictions)", ge=1, le=10)
    reasoning: str = Field(description="Concise explanation including: rule name, how accurately the response followed it, whether required links/phrases were used, and any deviations or omissions observed (2-3 sentences max)")


class OverallEvaluation(BaseModel):
    """Overall evaluation of agent performance"""
    tool_selection_score: float = Field(description="Score from 0-1 for tool selection accuracy", ge=1, le=10)
    tool_usage_score: float = Field(description="Score from 0-1 for correct tool parameter usage", ge=1, le=10)
    sop_adherence_score: float = Field(description="Score from 0-1 for SOP adherence", ge=1, le=10)
    custom_tone_score: float = Field(description="Score from 0-1 for custom tone adherence", ge=1, le=10)
    adjust_response_score: float = Field(description="Score from 0-1 for adjust response rule adherence", ge=1, le=10)
    response_accuracy_score: float = Field(description="Score from 0-1 for response accuracy", ge=1, le=10)
    improvement_suggestions: List[str] = Field(description="Specific suggestions for improvement")


class AgentEvaluationResult(BaseModel):
    """Complete evaluation result"""
    tool_evaluations: List[ToolInvocationEvaluation] = Field(description="Evaluation of each tool invocation")
    sop_evaluation: Optional[SOPEvaluation] = Field(description="Evaluation of SOP adherence if applicable")
    custom_tone_evaluation: CustomToneEvaluation = Field(description="Evaluation of custom tone adherence")
    adjust_response_evaluation: AdjustResponseEvaluation = Field(description="Evaluation of adjust response rules")
    response_evaluation: ResponseQualityEvaluation = Field(description="Evaluation of the final response")
    overall_evaluation: OverallEvaluation = Field(description="Overall performance assessment")


# Result generation helpers

def create_payload(requestId, rephrasedQuery, model_name, conversation_context_history=None):
    return {
        "botId": BOT_ID,
        "botRefId": BOT_REF_ID,
        "requestId": f"{model_name}_" + str(requestId),
        "messageId": "e5a1d9f6-8f7c-4203-9c2e-1f9e25cb3a89",
        "conversationId": f"{model_name}_conv",
        "rephrasedQuery": rephrasedQuery,
        "conversation_context_history": conversation_context_history or [],
        "filterTags": {
            "OR": [{"source_uuid": FILTER_TAGS_SOURCE_UUIDS}]
        },
        "llmName": LLM_NAME,
        "valid_sources_for_response_rules": FILTER_TAGS_SOURCE_UUIDS,
        "stream_direct": True,
        "sendFullAnswer": True,
        "evalRequest": True,
    }


def generate_response(response):
    return {
        "request_id": response["requestId"],
        "rephrase_query": response["nluInfo"]["rephrasedQuery"],
        "answer": response["nluInfo"]["generatedAnswer"],
        "answer_type": response["nluInfo"]["responseType"],
        "intermediate_steps": response["nluInfo"]["answerSourceDocuments"],
        "time_metrics": response["triggerIntent"],
        "conversation_history": response.get("conversation_history", []),
    }


def validate_latency_metrics(time_metrics) -> bool:
    """Validate that required latency metrics are present and non-zero."""
    if not time_metrics or len(time_metrics) == 0:
        return False
    metrics = time_metrics[0]
    for field in ("first_token_latency", "total_tool_latency"):
        if field not in metrics:
            return False
    has_last_token = "last_token_latency" in metrics and metrics["last_token_latency"] != 0
    has_llm_gen = "llm_generation_latency" in metrics and metrics["llm_generation_latency"] != 0
    return has_last_token or has_llm_gen


def run_conversation(convo_df, model_name):
    """Run all turns for a single conversation sequentially."""
    convo_id = convo_df["conversation_id"].iloc[0]
    convo_df = convo_df.sort_values("sequence")
    print(f"🧵 Running conversation {convo_id} with {len(convo_df)} turns")

    conversation_history: list = []
    row_results: list = []

    for _, row in convo_df.iterrows():
        requestId = row["requestId"]
        rephrasedQuery = row["rephrasedQuery"]
        payload = create_payload(
            requestId, rephrasedQuery, model_name,
            conversation_context_history=conversation_history,
        )
        try:
            response = requests.post(
                LOCAL_URL, json=payload, headers=HEADER, timeout=REQUEST_TIMEOUT
            )
            string_data = response.content.decode("utf-8")

            # SSE responses have multiple "data: {...}" lines — use the last valid JSON line
            result = None
            for line in reversed(string_data.splitlines()):
                line = line.strip()
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    try:
                        result = json.loads(payload)
                        break
                    except json.JSONDecodeError:
                        continue
                elif line:
                    try:
                        result = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if result is None:
                raise ValueError(f"No valid JSON found in response for {requestId}")

            if "conversationContextHistory" in result:
                updated_history = result["conversationContextHistory"]
                if len(updated_history) >= 2:
                    updated_history[-2]["content"] = rephrasedQuery
                conversation_history = updated_history
            result["conversation_history"] = conversation_history

            result_dict = generate_response(result)
            metadata = {
                "conversation_id": convo_id,
                "sequence": row.get("sequence"),
                "Lang": row.get("Lang"),
                "component": row.get("component"),
                "Expectation": row.get("Expectation"),
            }
            row_results.append({**metadata, **result_dict})

            if not validate_latency_metrics(result_dict.get("time_metrics")):
                print(f"⚠️  Skipping row {requestId} — latency metrics invalid")
                continue

        except Exception as exc:
            print(f"Failed for {row.get('requestId', 'unknown')} in conversation {convo_id}: {exc}")
            break

        time.sleep(SLEEP_TIMER)

    return row_results


def run_result_generation(dataset_path, model_name, output_dir):
    """Run the result generation phase."""
    print(f"Starting result generation for model: {model_name}")

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(f"{model_output_dir}/pickle_files", exist_ok=True)

    eval_dataset = pd.read_pickle(dataset_path)
    eval_dataset = eval_dataset[:1]
    eval_dataset.reset_index(drop=True, inplace=True)
    eval_dataset.sort_values(by=["conversation_id", "sequence"], inplace=True)
    eval_dataset.reset_index(drop=True, inplace=True)

    print(f"Loaded dataset with {len(eval_dataset)} records")

    all_results: list = []
    for convo_id, convo_df in eval_dataset.groupby("conversation_id", group_keys=False):
        print(f"\n=== Processing conversation {convo_id} ({len(convo_df)} rows) ===")
        all_results.extend(run_conversation(convo_df, model_name))

    valid_results = len(all_results)
    invalid_results = len(eval_dataset) - valid_results
    final_output_path = f"{model_output_dir}/agent_using_{model_name}_FINAL.pkl"
    with open(final_output_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Result generation completed. Saved to: {final_output_path}")
    print(f"Summary: {valid_results} valid, {invalid_results} invalid/failed")
    return final_output_path


# Evaluation helpers

class AgentEvaluator:
    def __init__(
        self,
        llm: Optional[Any] = None,
        available_sop: str = AVAILABLE_SOP,
        custom_tone_instructions: str = CUSTOM_TONE_INSTRUCTIONS,
        adjust_response_rules: str = ADJUST_RESPONSE_RULES,
        eval_prompt: str = EVALUATION_PROMPT,
    ):
        """
        Initialize the evaluator with an LLM instance and the bot-scoped config
        content (SOPs, custom tone instructions, adjust-response rules) and the
        evaluation prompt template, all of which default to the module-level
        values loaded from config.yaml / eval_utils.py but can be overridden
        per call (e.g. to evaluate a different bot in the same process).
        """
        self.llm = llm
        self.available_sop = available_sop
        self.custom_tone_instructions = custom_tone_instructions
        self.adjust_response_rules = adjust_response_rules
        self.output_parser = PydanticOutputParser(pydantic_object=AgentEvaluationResult)
        self.prompt = ChatPromptTemplate.from_template(eval_prompt)

    def format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool descriptions for the prompt."""
        formatted_tools = []
        for tool in tools:
            tool_info = f"""
Tool: {tool.get('tool_name', 'Unknown')}
Title: {tool.get('action_title', 'N/A')}
Description: {tool.get('description', 'N/A')}
Input Schema: {json.dumps(tool.get('input_schema', {}), indent=2)}
Output Schema: {json.dumps(tool.get('output_schema', {}), indent=2)}
"""
            formatted_tools.append(tool_info)
        return "\n" + "=" * 50 + "\n".join(formatted_tools)

    def format_intermediate_steps(self, steps: List[Dict[str, Any]]) -> str:
        """Format intermediate steps for the prompt."""
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            step_info = f"""
Step {i}:
- Tool: {step.get('tool', 'Unknown')}
- Action Title: {step.get('action_title', 'N/A')}
- Tool Input: {json.dumps(step.get('tool_input', {}), indent=2)}
- Tool Output: {json.dumps(step.get('tool_output', {}), indent=2)}
"""
            formatted_steps.append(step_info)
        return "\n" + "-" * 30 + "\n".join(formatted_steps)

    def evaluate(
        self,
        conversation_id: str,
        sequence_number: int,
        component: str,
        available_tools: List[Dict[str, Any]],
        user_query: str,
        conversation_history: List[Dict[str, Any]],
        intermediate_steps: List[Dict[str, Any]],
        final_response: str,
    ) -> tuple:
        """Evaluate agent performance."""
        formatted_tools = self.format_tools(available_tools)
        formatted_steps = self.format_intermediate_steps(intermediate_steps)

        messages = self.prompt.format_messages(
            conversation_id=conversation_id,
            sequence_number=sequence_number,
            component=component,
            available_tools=formatted_tools,
            user_query=user_query,
            conversation_history=conversation_history,
            available_sops=self.available_sop,
            custom_tone_instructions=self.custom_tone_instructions,
            adjust_response_rules=self.adjust_response_rules,
            intermediate_steps=formatted_steps,
            final_response=final_response,
            format_instructions=self.output_parser.get_format_instructions(),
        )

        response = self.llm.invoke(messages)

        # Handle both string and list content formats
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "".join(text_parts) if text_parts else str(content)

        try:
            evaluation_result = self.output_parser.parse(content)
            return user_query, evaluation_result
        except Exception as exc:
            error_msg = str(exc)
            if "validation errors" in error_msg.lower() or "Field required" in error_msg:
                print(f"Error parsing evaluation result: {error_msg}")
                print("The LLM response is missing required fields.")
            else:
                print(f"Error parsing evaluation result: {error_msg}")
            print(f"Raw response content: {str(content)}")
            raise


def clean_dict_for_pickle(obj):
    """Recursively clean a dict/list to ensure only picklable types."""
    if isinstance(obj, dict):
        return {k: clean_dict_for_pickle(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_for_pickle(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, "model_dump"):   # Pydantic v2
        return clean_dict_for_pickle(obj.model_dump())
    elif hasattr(obj, "dict"):         # Pydantic v1
        return clean_dict_for_pickle(obj.dict())
    else:
        return str(obj)


def extract_evaluation_to_dict(user_query, evaluation_result: AgentEvaluationResult) -> Dict[str, Any]:
    """Extract all values from AgentEvaluationResult into a flat dictionary."""
    result_dict: Dict[str, Any] = {"user_query": user_query}

    overall = evaluation_result.overall_evaluation
    result_dict.update({
        "tool_selection_score": overall.tool_selection_score,
        "tool_usage_score": overall.tool_usage_score,
        "sop_adherence_score": overall.sop_adherence_score,
        "custom_tone_score": overall.custom_tone_score,
        "adjust_response_score": overall.adjust_response_score,
        "response_accuracy_score": overall.response_accuracy_score,
        "improvement_suggestions": overall.improvement_suggestions,
        "improvement_suggestions_count": len(overall.improvement_suggestions),
    })

    scores = [
        overall.tool_selection_score,
        overall.tool_usage_score,
        overall.sop_adherence_score,
        overall.custom_tone_score,
        overall.adjust_response_score,
        overall.response_accuracy_score,
    ]
    result_dict["overall_score"] = sum(scores) / len(scores) if scores else 0.0

    if evaluation_result.sop_evaluation is not None:
        sop_eval = evaluation_result.sop_evaluation
        result_dict.update({
            "sop_identified": sop_eval.sop_identified,
            "sop_trigger_accuracy": sop_eval.sop_trigger_accuracy,
            "sop_execution_accuracy": sop_eval.sop_execution_accuracy,
            "sop_reasoning": sop_eval.reasoning,
        })
    else:
        result_dict.update({
            "sop_identified": None,
            "sop_trigger_accuracy": None,
            "sop_execution_accuracy": None,
            "sop_reasoning": None,
        })

    custom_tone_eval = evaluation_result.custom_tone_evaluation
    result_dict.update({
        "brand_voice_score": custom_tone_eval.brand_voice_score,
        "terminology_score": custom_tone_eval.terminology_score,
        "custom_tone_reasoning": custom_tone_eval.reasoning,
    })

    adj_eval = evaluation_result.adjust_response_evaluation
    result_dict.update({
        "adjust_response_rule_identified": adj_eval.rule_identified,
        "adjust_response_rule_trigger_accuracy": adj_eval.rule_trigger_accuracy,
        "adjust_response_rule_execution_accuracy": adj_eval.rule_execution_accuracy,
        "adjust_response_reasoning": adj_eval.reasoning,
    })

    resp_eval = evaluation_result.response_evaluation
    result_dict.update({
        "addresses_user_query": resp_eval.addresses_user_query,
        "uses_tool_output_correctly": resp_eval.uses_tool_output_correctly,
        "has_hallucination": resp_eval.has_hallucination,
        "is_complete": resp_eval.is_complete,
        "hallucination_details": resp_eval.hallucination_details,
    })

    result_dict["total_tools_used"] = len(evaluation_result.tool_evaluations)
    tool_results = []
    for i, tool_eval in enumerate(evaluation_result.tool_evaluations):
        prefix = f"tool_{i + 1}_" if len(evaluation_result.tool_evaluations) > 1 else "tool_"
        tool_results.append({
            f"{prefix}name": tool_eval.tool_name,
            f"{prefix}is_correct_tool": tool_eval.is_correct_tool,
            f"{prefix}input_parameters_correct": tool_eval.input_parameters_correct,
            f"{prefix}reasoning": tool_eval.reasoning,
        })
    result_dict["tool_wise_results"] = tool_results

    if evaluation_result.tool_evaluations:
        result_dict["all_tools_correct"] = all(te.is_correct_tool for te in evaluation_result.tool_evaluations)
        result_dict["all_parameters_correct"] = all(te.input_parameters_correct for te in evaluation_result.tool_evaluations)
        result_dict["correct_tools_count"] = sum(1 for te in evaluation_result.tool_evaluations if te.is_correct_tool)
        result_dict["correct_parameters_count"] = sum(1 for te in evaluation_result.tool_evaluations if te.input_parameters_correct)
    else:
        result_dict.update({
            "all_tools_correct": False,
            "all_parameters_correct": False,
            "correct_tools_count": 0,
            "correct_parameters_count": 0,
        })

    result_dict["is_high_quality"] = (
        not result_dict["has_hallucination"]
        and result_dict["addresses_user_query"]
        and result_dict["is_complete"]
    )
    result_dict["needs_improvement"] = (
        result_dict["has_hallucination"] or not result_dict["addresses_user_query"]
    )

    return result_dict


def safe_get_latency(metrics, key, default=0):
    """Safely retrieve a latency key from the metrics structure."""
    try:
        if isinstance(metrics, list) and metrics and isinstance(metrics[0], dict):
            return metrics[0].get(key, default)
        elif isinstance(metrics, dict):
            return metrics.get(key, default)
    except Exception:
        pass
    return default


def load_existing_batch_results(batch_results_dir):
    """Load all existing batch result pickles; return (evaluated_ids, results, highest_batch_num)."""
    evaluated_request_ids: set = set()
    existing_batch_results: list = []
    highest_batch_num = 0

    if not os.path.exists(batch_results_dir):
        return evaluated_request_ids, existing_batch_results, highest_batch_num

    batch_files = sorted(
        [f for f in os.listdir(batch_results_dir) if f.endswith(".pkl") and f.startswith("batch_")]
    )
    for batch_file in batch_files:
        batch_path = os.path.join(batch_results_dir, batch_file)
        try:
            batch_num_str = batch_file.replace("batch_", "").replace(".pkl", "")
            try:
                batch_num = int(batch_num_str)
                highest_batch_num = max(highest_batch_num, batch_num)
            except ValueError:
                print(f"Warning: Could not parse batch number from {batch_file}")
            with open(batch_path, "rb") as f:
                batch_data = pickle.load(f)
                existing_batch_results.extend(batch_data)
                for item in batch_data:
                    if isinstance(item, dict) and "request_id" in item:
                        evaluated_request_ids.add(item["request_id"])
        except Exception as exc:
            print(f"Warning: Could not load batch file {batch_file}: {exc}")

    return evaluated_request_ids, existing_batch_results, highest_batch_num


def run_evaluation(results_file_path, model_name, output_dir):
    """Run the evaluation phase."""
    print(f"Starting evaluation for model: {model_name}")

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    batch_results_dir = os.path.join(model_output_dir, "batch_results")
    os.makedirs(batch_results_dir, exist_ok=True)

    with open(results_file_path, "rb") as f:
        results_data = pickle.load(f)
    results_df = pd.DataFrame(results_data)
    print(f"Loaded {len(results_df)} results for evaluation")

    evaluated_request_ids, existing_batch_results, highest_batch_num = load_existing_batch_results(
        batch_results_dir
    )
    if evaluated_request_ids:
        print(f"Found {len(evaluated_request_ids)} already-evaluated request IDs — filtering …")
        initial_count = len(results_df)
        results_df = results_df[~results_df["request_id"].isin(evaluated_request_ids)]
        print(
            f"Filtered to {len(results_df)} remaining records "
            f"(skipped {initial_count - len(results_df)} already evaluated)"
        )
        print(f"Resuming from batch {highest_batch_num + 1}")
    else:
        print("No existing batch results found. Starting fresh evaluation.")
        highest_batch_num = 0

    reasoning = {"effort": EVAL_REASONING_EFFORT}

    def evaluate_row(row):
        """Evaluate a single result row — runs in parallel worker processes."""
        try:
            if not row.get("rephrase_query") or not row.get("answer") or row.get("answer") == "":
                print(f"[evaluate_row] Skipping {row.get('request_id')} — empty rephrase_query or answer")
                return None

            # Create a fresh evaluator per worker to avoid pickling SSLContext
            llm_eval = ChatOpenAI(model_name=EVAL_MODEL_NAME, reasoning=reasoning)
            evaluator = AgentEvaluator(
                llm_eval,
                available_sop=AVAILABLE_SOP,
                custom_tone_instructions=CUSTOM_TONE_INSTRUCTIONS,
                adjust_response_rules=ADJUST_RESPONSE_RULES,
                eval_prompt=EVALUATION_PROMPT,
            )

            user_query, result = evaluator.evaluate(
                conversation_id=row["conversation_id"],
                sequence_number=row["sequence"],
                component=row["component"],
                available_tools=TOOL_DATASET,
                user_query=row["rephrase_query"],
                intermediate_steps=row.get("intermediate_steps", []),
                final_response=row["answer"],
                conversation_history=row.get("conversation_history", []),
            )

            overall = result.overall_evaluation
            scores = [
                overall.tool_selection_score,
                overall.tool_usage_score,
                overall.sop_adherence_score,
                overall.custom_tone_score,
                overall.adjust_response_score,
                overall.response_accuracy_score,
            ]
            overall_score = sum(scores) / len(scores) if scores else 0.0
            print(
                f"Evaluation {row['conversation_id']} - {row['sequence']} "
                f"(request_id: {row['request_id']})  Overall: {overall_score:.2f}"
            )
            return (row["request_id"], user_query, result)
        except Exception as exc:
            import traceback
            print(f"Error evaluating row {row.get('request_id', row.name)}: {exc}")
            traceback.print_exc()
            return None

    BATCH_SIZE = EVAL_BATCH_SIZE
    DELAY_SECONDS = EVAL_DELAY_SECONDS
    total_rows = len(results_df)
    all_parallel_results: list = []

    if total_rows == 0:
        print("No records to evaluate — all already processed.")
    else:
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Running parallel evaluation: {total_rows} records in {num_batches} batches of {BATCH_SIZE}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            batch_df = results_df.iloc[start_idx:end_idx]
            actual_batch_num = highest_batch_num + batch_idx + 1

            print(f"\nProcessing batch {actual_batch_num} (rows {start_idx + 1}–{end_idx}) …")
            print(f"  Batch columns: {list(batch_df.columns)}")
            print(f"  First row sample: request_id={batch_df.iloc[0].get('request_id')}, answer_len={len(str(batch_df.iloc[0].get('answer', '') or ''))}")
            # Use sequential apply when batch is small to surface hidden worker errors
            if len(batch_df) <= 3:
                batch_results = batch_df.apply(evaluate_row, axis=1)
            else:
                batch_results = batch_df.parallel_apply(evaluate_row, axis=1)
            batch_results = [res for res in batch_results if res is not None]
            all_parallel_results.extend(batch_results)

            batch_evaluation_dicts = []
            for request_id, user_query, result in batch_results:
                eval_dict = extract_evaluation_to_dict(user_query, result)
                eval_dict["request_id"] = request_id
                eval_dict = clean_dict_for_pickle(eval_dict)
                batch_evaluation_dicts.append(eval_dict)

            batch_file_path = os.path.join(batch_results_dir, f"batch_{actual_batch_num:04d}.pkl")
            with open(batch_file_path, "wb") as f:
                pickle.dump(batch_evaluation_dicts, f)
            print(f"Batch {actual_batch_num} done: {len(batch_results)} evaluations → {batch_file_path}")

            if batch_idx < num_batches - 1:
                print(f"Waiting {DELAY_SECONDS}s before next batch …")
                time.sleep(DELAY_SECONDS)

        print(f"\nAll batches done: {len(all_parallel_results)} / {total_rows} successful")

    # Combine existing + new
    if existing_batch_results:
        new_dicts = []
        for request_id, user_query, result in all_parallel_results:
            ed = extract_evaluation_to_dict(user_query, result)
            ed["request_id"] = request_id
            new_dicts.append(clean_dict_for_pickle(ed))
        all_evaluation_dicts = existing_batch_results + new_dicts
    else:
        all_evaluation_dicts = []
        for request_id, user_query, result in all_parallel_results:
            ed = extract_evaluation_to_dict(user_query, result)
            ed["request_id"] = request_id
            all_evaluation_dicts.append(clean_dict_for_pickle(ed))

    evaluation_dataset = clean_dict_for_pickle(all_evaluation_dicts)

    eval_results_path = f"{model_output_dir}/{model_name}_evaluation_results.pkl"
    with open(eval_results_path, "wb") as f:
        pickle.dump(evaluation_dataset, f)

    eval_results_df = pd.DataFrame(evaluation_dataset)

    with open(results_file_path, "rb") as f:
        original_results_data = pickle.load(f)
    original_results_df = pd.DataFrame(original_results_data)
    original_results_df.rename(columns={"rephrase_query": "user_query"}, inplace=True)

    combined_df = pd.DataFrame()
    if not eval_results_df.empty and not original_results_df.empty:
        expected_cols = [
            "request_id", "conversation_id", "sequence", "component", "Lang",
            "user_query", "conversation_history", "intermediate_steps", "answer", "time_metrics",
        ]
        # Filter existing columns in original_results_df to prevent KeyError
        existing_cols = [col for col in expected_cols if col in original_results_df.columns]
        
        combined_df = eval_results_df.merge(
            original_results_df[existing_cols],
            on=["user_query", "request_id"],
            how="inner",
        )

    if "time_metrics" in combined_df.columns:
        try:
            combined_df["first_token_llm_latency"] = combined_df["time_metrics"].apply(
                lambda x: safe_get_latency(x, "first_token_latency") - safe_get_latency(x, "total_tool_latency")
            )
            combined_df["last_token_llm_latency"] = combined_df["time_metrics"].apply(
                lambda x: (
                    safe_get_latency(x, "last_token_latency") - safe_get_latency(x, "total_tool_latency")
                    if safe_get_latency(x, "last_token_latency") != 0
                    else safe_get_latency(x, "llm_generation_latency") - safe_get_latency(x, "total_tool_latency")
                )
            )
        except Exception as exc:
            print(f"Warning: Could not calculate latency metrics: {exc}")

    final_columns = [
        "request_id", "conversation_id", "sequence", "component", "Lang",
        "user_query", "conversation_history", "intermediate_steps", "answer",
        "tool_selection_score", "tool_usage_score", "sop_adherence_score",
        "custom_tone_score", "adjust_response_score", "response_accuracy_score", "overall_score",
        "improvement_suggestions", "improvement_suggestions_count",
        "sop_identified", "sop_trigger_accuracy", "sop_execution_accuracy", "sop_reasoning",
        "brand_voice_score", "terminology_score", "custom_tone_reasoning",
        "adjust_response_rule_identified", "adjust_response_rule_trigger_accuracy",
        "adjust_response_rule_execution_accuracy", "adjust_response_reasoning",
        "addresses_user_query", "uses_tool_output_correctly", "has_hallucination",
        "is_complete", "hallucination_details",
        "total_tools_used", "tool_wise_results",
    ]
    if "first_token_llm_latency" in combined_df.columns:
        final_columns.extend(["first_token_llm_latency", "last_token_llm_latency"])

    final_df = pd.DataFrame()
    if not combined_df.empty:
        available_columns = [col for col in final_columns if col in combined_df.columns]
        final_df = combined_df[available_columns]
        final_df.sort_values(by=["component", "Lang", "conversation_id", "sequence"], inplace=True)

    csv_output_path = f"{model_output_dir}/{model_name}_evaluation_results.csv"
    final_df.to_csv(csv_output_path, index=False)

    print(f"\nEvaluation completed. Results saved to:")
    print(f"  Pickle : {eval_results_path}")
    print(f"  CSV    : {csv_output_path}")

    print(f"\nEvaluation Summary:")
    print(f"  Total evaluations    : {len(eval_results_df)}")
    if not eval_results_df.empty:
        print(f"  Mean overall score   : {eval_results_df['overall_score'].mean():.3f}")
        print(f"  Mean tool selection  : {eval_results_df['tool_selection_score'].mean():.3f}")
        print(f"  Mean tool usage      : {eval_results_df['tool_usage_score'].mean():.3f}")
        print(f"  Mean SOP adherence   : {eval_results_df['sop_adherence_score'].mean():.3f}")
        print(f"  Mean custom tone     : {eval_results_df['custom_tone_score'].mean():.3f}")
        print(f"  Mean adjust response : {eval_results_df['adjust_response_score'].mean():.3f}")
        print(f"  Mean resp. accuracy  : {eval_results_df['response_accuracy_score'].mean():.3f}")
        print(f"  Hallucinations       : {eval_results_df['has_hallucination'].sum()}")
    else:
        print("  No evaluations were processed.")

    return csv_output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point.

    All CLI flags are optional — values fall back to the ``defaults`` block
    in config.yaml when not provided.  A flag on the command line always
    takes priority over the config value.
    """
    parser = argparse.ArgumentParser(
        description="llm-eval-framework — Combined Result Generation and Evaluation"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME,
        help=f"Model name to evaluate (config.yaml default: {DEFAULT_MODEL_NAME!r})",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to the input dataset")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    parser.add_argument(
        "--skip_generation", action="store_true", default=DEFAULT_SKIP_GENERATION,
        help="Skip result generation and only run evaluation",
    )
    parser.add_argument(
        "--skip_evaluation", action="store_true", default=DEFAULT_SKIP_EVALUATION,
        help="Skip result evaluation and only run generation",
    )
    parser.add_argument(
        "--results_file", default=DEFAULT_RESULTS_FILE,
        help="Path to existing results file (required with --skip_generation)",
    )

    args = parser.parse_args()

    if not args.model:
        print("Error: --model was not provided and no defaults.model_name is set in config.yaml")
        return

    print(f"Starting combined evaluation for model : {args.model}")
    print(f"Base output directory                  : {args.output_dir}")
    print(f"Model-specific directory               : {args.output_dir}/{args.model}")

    if args.skip_generation:
        if not args.results_file:
            print("Error: --results_file is required when --skip_generation is used")
            return
        results_file_path = args.results_file
    else:
        results_file_path = run_result_generation(args.dataset, args.model, args.output_dir)

    if args.skip_evaluation:
        print("Result generation completed successfully.")
        return

    final_output_path = run_evaluation(results_file_path, args.model, args.output_dir)
    print(f"\nCombined evaluation completed successfully!")
    print(f"Final results: {final_output_path}")


if __name__ == "__main__":
    main()
