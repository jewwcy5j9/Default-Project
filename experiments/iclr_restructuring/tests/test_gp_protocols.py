import sys
from pathlib import Path

from sklearn.gaussian_process.kernels import ConstantKernel, Product, RBF, Sum, WhiteKernel


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gp_protocols import (
    PRIMARY_GP_PROTOCOL,
    T5_DIAGNOSTIC_GP_PROTOCOL,
    make_primary_gp,
    make_t5_diagnostic_gp,
)


def test_primary_gp_is_locked_and_fresh():
    first = make_primary_gp()
    second = make_primary_gp()
    assert first is not second and first.kernel is not second.kernel
    assert isinstance(first.kernel, Sum)
    assert isinstance(first.kernel.k1, Product)
    assert isinstance(first.kernel.k1.k1, ConstantKernel)
    assert isinstance(first.kernel.k1.k2, RBF)
    assert isinstance(first.kernel.k2, WhiteKernel)
    assert first.kernel.k1.k1.constant_value == 1.0
    assert first.kernel.k1.k2.length_scale == 1.0
    assert first.kernel.k2.noise_level == 0.05
    assert first.alpha == 1e-4
    assert first.normalize_y is True
    assert first.n_restarts_optimizer == 1
    assert first.random_state == 0
    assert PRIMARY_GP_PROTOCOL["id"] == "GP-primary-0.05-v1"


def test_t5_diagnostic_is_explicitly_distinct():
    primary = make_primary_gp()
    diagnostic = make_t5_diagnostic_gp()
    assert isinstance(diagnostic.kernel, Sum)
    assert isinstance(diagnostic.kernel.k1, RBF)
    assert isinstance(diagnostic.kernel.k2, WhiteKernel)
    assert diagnostic.kernel.k2.noise_level == 0.01
    assert diagnostic.n_restarts_optimizer == 0
    assert str(diagnostic.kernel) != str(primary.kernel)
    assert T5_DIAGNOSTIC_GP_PROTOCOL["display_name"] == "GP-T5-diagnostic"


def test_k3_followup_uses_unified_factory_and_diagnostic_label():
    source = (HERE.parent / "k3_followup.py").read_text(encoding="utf-8")
    assert "make_primary_gp()" in source
    assert "GaussianProcessRegressor(" not in source
    assert "GPR-probability-diagnostic" in source
