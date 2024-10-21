from pathlib import Path

from testbook import testbook

notebooks_dir = Path(__file__).parents[1] / "notebooks"


@testbook(f"{notebooks_dir / 'dti_preprocessing'}.ipynb", execute=True)
def test_func(tb):
    s4l_dti_file: Path = tb.ref("s4l_dti_file")

    assert s4l_dti_file.exists()
