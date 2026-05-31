from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    product_idea: str = Field(..., description="产品想法描述", min_length=1)
    supplementary_info: str = Field(default="", description="补充信息")
    selected_model: str | None = Field(default=None, description="指定模型，不填则自动选择")
    temperature: float | None = Field(default=None, ge=0.0, le=1.0, description="LLM 温度")
    reflection_max_rounds: int | None = Field(default=None, ge=0, le=3, description="最大反思轮数")
    reviewer_score_threshold: int | None = Field(default=None, ge=0, le=100, description="评审通过阈值")


class RevisionRequest(BaseModel):
    existing_state: dict = Field(..., description="完整的现有 state")
    feedback: str = Field(..., description="修订意见", min_length=1)
    selected_model: str | None = Field(default=None)
