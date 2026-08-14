from scripts.run_vertex_physical_transfer_ceiling_design_v1 import build_prompt


def test_prompt_covers_sparse_ceiling_transfer_question_without_private_rows() -> None:
    prompt = build_prompt()
    assert "22 safely linked athlete profiles" in prompt
    assert "50%-flash Kilter-equivalent grade" in prompt
    assert "hierarchical stochastic-frontier/ceiling" in prompt
    assert "ordinal cumulative-link models" in prompt
    assert "Missing means unobserved, not weak" in prompt
    assert "athlete names" not in prompt.lower()
    assert "no private rows are transmitted" in prompt.lower()
