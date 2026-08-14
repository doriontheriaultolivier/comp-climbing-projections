from pathlib import Path

import pandas as pd

from scripts.audit_pathway_hierarchy_v1 import run


def test_hierarchy_audit_binds_support_and_remains_unfitted(tmp_path: Path) -> None:
    support = tmp_path / "support.csv"
    pd.DataFrame({
        "pathway_target_domain": ["other", "cont_africa", "wc"],
        "direct_event_contexts": [100, 2, 70],
        "wc_bridge_fraction": [0.1, 0.03, 1.0],
    }).to_csv(support, index=False)
    output = tmp_path / "hierarchy.csv"
    receipt_path = tmp_path / "receipt.json"
    receipt = run(support, output, receipt_path)
    hierarchy = pd.read_csv(output)
    assert receipt["status"] == "UNFITTED_HIERARCHY_RESEARCH_ONLY"
    assert not receipt["hyperparameters_selected"]
    assert set(hierarchy["pathway_target_domain"]) == {"cont_africa", "wc"}
    assert not hierarchy["selected_for_model"].any()
