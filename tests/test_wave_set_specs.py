"""load_wave_set_specs's m_sum = m_p + m_q validator: valid config loads,
invalid one raises naming the role key and triad."""
import textwrap

import pytest

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs

_VALID = textwrap.dedent("""\
    quartet_ok:
      h_e: 10000
      modes:
        a: {m: 4, n: 5, alpha: 3, u: 30.0}
        b: {m: 3, n: 4, alpha: 3, u: 30.0}
        c: {m: 1, n: 2, alpha: 3, u: 30.0}
      triads:
        - {sum: a, members: [b, c], display_label: "Triad 1", triad_key: null}
      reference_triad: 0
    """)

_INVALID = textwrap.dedent("""\
    quartet_bad:
      h_e: 10000
      modes:
        a: {m: 4, n: 5, alpha: 3, u: 30.0}
        b: {m: 3, n: 4, alpha: 3, u: 30.0}
        c: {m: 2, n: 9, alpha: 3, u: 30.0}
      triads:
        - {sum: a, members: [b, c], display_label: "Triad 1", triad_key: null}
      reference_triad: 0
    """)


def test_valid_wave_set_loads(tmp_path):
    path = tmp_path / "specs.yaml"
    path.write_text(_VALID)
    specs = load_wave_set_specs(str(path))
    assert set(specs) == {"quartet_ok"}
    assert specs["quartet_ok"].n_modes() == 3


def test_invalid_m_sum_raises_naming_key_and_triad(tmp_path):
    path = tmp_path / "specs.yaml"
    path.write_text(_INVALID)
    with pytest.raises(ValueError, match=r"quartet_bad.*sum=a.*members=.*m_sum=4 != m_p\+m_q=3\+2=5"):
        load_wave_set_specs(str(path))
