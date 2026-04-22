from __future__ import annotations

from portfolio_toolkit import build_features, load_prices, make_forward_alpha_target, make_forward_realized_vol_target, make_forward_return_target


def test_feature_builder_and_targets_smoke(repo_root):
    prices = load_prices("shared_set_1", repo_root=repo_root)
    features = build_features(prices)
    assert {"date", "ticker", "return_5d", "vol_20d", "momentum_20d", "beta_20d_spy"}.issubset(features.columns)

    forward_return = make_forward_return_target(prices, 5)
    forward_alpha = make_forward_alpha_target(prices, 5)
    forward_vol = make_forward_realized_vol_target(prices, 5)
    assert "forward_return_5d" in forward_return.columns
    assert "forward_alpha_5d_vs_spy" in forward_alpha.columns
    assert "forward_realized_vol_5d" in forward_vol.columns
