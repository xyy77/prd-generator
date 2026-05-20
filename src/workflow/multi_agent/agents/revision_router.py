from src.utils.logger import get_logger

logger = get_logger(__name__)


def revision_router_node(state: dict, reference_context: str = "") -> dict:
    """Determine which agents need revision based on reviewer feedback.

    Sets agents_to_revise and increments reflection_round.
    If no agents flagged (all null feedback), treats as finalize.
    """
    feedback = state.get("reviewer_feedback", {})
    current_round = state.get("reflection_round", 0)
    max_rounds = state.get("reflection_max_rounds", 2)

    agents_to_revise = [name for name, fb in feedback.items() if fb and str(fb).lower() != "null"]

    history_entry = {
        "round": current_round + 1,
        "score": state.get("reviewer_score", 0),
        "feedback": feedback,
        "agents_to_revise": agents_to_revise,
    }
    history = list(state.get("reflection_history", []))
    history.append(history_entry)

    if not agents_to_revise or current_round >= max_rounds - 1:
        logger.info("No agents to revise or max rounds reached, finalizing")
        return {
            "reflection_round": current_round + 1,
            "reflection_history": history,
            "agents_to_revise": [],
            "current_stage": "revision_router_finalize",
        }

    logger.info("Reflection round %d: revising agents %s", current_round + 1, agents_to_revise)
    return {
        "reflection_round": current_round + 1,
        "reflection_history": history,
        "agents_to_revise": agents_to_revise,
        "current_stage": "revision_router_revise",
    }
