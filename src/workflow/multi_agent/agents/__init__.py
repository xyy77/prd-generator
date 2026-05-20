from src.workflow.multi_agent.agents.requirements_analyst import requirements_analyst_node
from src.workflow.multi_agent.agents.feature_planner import feature_planner_node
from src.workflow.multi_agent.agents.ux_designer import ux_designer_node
from src.workflow.multi_agent.agents.tech_advisor import tech_advisor_node
from src.workflow.multi_agent.agents.reviewer import reviewer_node
from src.workflow.multi_agent.agents.image_analyst import image_analyst_node
from src.workflow.multi_agent.agents.revision_router import revision_router_node

AGENT_NODE_MAP = {
    "requirements_analyst": requirements_analyst_node,
    "feature_planner": feature_planner_node,
    "ux_designer": ux_designer_node,
    "tech_advisor": tech_advisor_node,
}
