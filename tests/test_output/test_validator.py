from src.output.validator import validate_prd_json, repair_prd_json, parse_and_validate


class TestValidatePrdJson:
    def test_valid_json_passes(self):
        data = {
            "version_record": {},
            "background_and_goals": {},
            "user_personas": [],
            "functional_requirements": [],
            "non_functional_requirements": {},
            "tech_architecture": {},
            "analytics_and_iteration": {},
            "risks_and_mitigation": [],
            "appendix": {},
        }
        valid, errors = validate_prd_json(data)
        assert valid
        assert errors == []

    def test_missing_fields_detected(self):
        data = {"version_record": {}}
        valid, errors = validate_prd_json(data)
        assert not valid
        assert len(errors) > 0


class TestRepairPrdJson:
    def test_fills_missing_fields(self):
        data = {"version_record": {"product_name": "Test"}}
        repaired = repair_prd_json(data)
        for field in [
            "background_and_goals",
            "user_personas",
            "functional_requirements",
        ]:
            assert field in repaired


class TestParseAndValidate:
    def test_clean_json(self):
        raw = '{"version_record":{},"background_and_goals":{},"user_personas":[],"functional_requirements":[],"non_functional_requirements":{},"tech_architecture":{},"analytics_and_iteration":{},"risks_and_mitigation":[],"appendix":{}}'
        data = parse_and_validate(raw)
        assert "version_record" in data

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"version_record":{},"background_and_goals":{},"user_personas":[],"functional_requirements":[],"non_functional_requirements":{},"tech_architecture":{},"analytics_and_iteration":{},"risks_and_mitigation":[],"appendix":{}}\n```'
        data = parse_and_validate(raw)
        assert "version_record" in data
