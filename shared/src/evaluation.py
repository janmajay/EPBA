import asyncio
import os
from langfuse import Langfuse
from shared.src.config import settings
from shared.src.logger import configure_logger

logger = configure_logger("evaluation")

# --- Heavy evaluation dependencies moved to top to avoid GIL lock delay during threading ---
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
# ----------------------------------------------------------------------------------------

# Langfuse client
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST
)

def _update_langfuse_trace(trace_id: str, scores: dict):
    """Utility to patch a Langfuse trace with metric scores."""
    import math
    try:
        for metric_name, value in scores.items():
            try:
                if isinstance(value, dict):
                    float_val = float(value.get("value", 0.0))
                    comment = value.get("reason", "")
                else:
                    float_val = float(value)
                    comment = ""
                    
                if math.isnan(float_val):
                    continue
                langfuse.score(
                    trace_id=trace_id,
                    name=metric_name,
                    value=float_val,
                    comment=comment
                )
            except (ValueError, TypeError):
                continue
        langfuse.flush()
        logger.info("patched_langfuse_trace", trace_id=trace_id, scores=scores)
    except Exception as e:
        import traceback
        logger.error("failed_to_patch_langfuse", trace_id=trace_id, error=str(e), traceback=traceback.format_exc())

def run_deepeval_retrieval(query: str, response: str, contexts: list[str], trace_id: str):
    """
    Evaluates retrieval quality using DeepEval (gpt-4o) in a background thread.
    """
    if not trace_id:
        logger.warning("no_trace_id_for_retrieval_eval")
        return

    logger.info("starting_deepeval_retrieval", trace_id=trace_id)
    try:
        evaluator_model = "gpt-4o"
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=contexts
        )

        context_relevance = ContextualRelevancyMetric(threshold=0.5, model=evaluator_model, include_reason=True)
        context_relevance.measure(test_case)

        scores = {
            "Contextual Relevancy": {
                "value": context_relevance.score,
                "reason": context_relevance.reason
            }
        }

        # Push to Langfuse
        _update_langfuse_trace(trace_id, scores)
        
    except Exception as e:
        import traceback
        logger.error("retrieval_eval_failed", trace_id=trace_id, error=str(e), traceback=traceback.format_exc())


def run_deepeval(query: str, response: str, contexts: list[str], trace_id: str):
    """
    Evaluates hallucinaton using DeepEval (gpt-4o) in a background thread.
    """
    if not trace_id:
        logger.warning("no_trace_id_for_deepeval")
        return

    logger.info("starting_deepeval", trace_id=trace_id)
    try:
        # Force GPT-4o evaluator
        evaluator_model = "gpt-4o"

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=contexts
        )

        faithfulness = FaithfulnessMetric(threshold=0.5, model=evaluator_model, include_reason=True)
        answer_relevance = AnswerRelevancyMetric(threshold=0.5, model=evaluator_model, include_reason=True)

        faithfulness.measure(test_case)
        answer_relevance.measure(test_case)

        scores = {
            "Faithfulness": {
                "value": faithfulness.score,
                "reason": faithfulness.reason
            },
            "Answer Relevance": {
                "value": answer_relevance.score,
                "reason": answer_relevance.reason
            }
        }

        # Push to Langfuse
        _update_langfuse_trace(trace_id, scores)

    except Exception as e:
        logger.error("deepeval_failed", trace_id=trace_id, error=str(e))
