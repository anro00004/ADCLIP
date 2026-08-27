import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def pytest_addoption(parser):
    """
    run_full_corpus flag: 
    True: Slower test because it checks by aligning with Muscle(--super 5) the whole corpus.
    False (default): Skip this test.
    """
    parser.addoption("--run_full_corpus", action="store_true", default=False,
                      help="run full corpus test with alignment that takes more time")
