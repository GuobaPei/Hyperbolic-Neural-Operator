from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]).read_bytes()
folder = root / "paper-source"
folder.mkdir(exist_ok=True)
for part in folder.glob("paper.part*"):
    part.unlink()
size = 512 * 1024
for number, start in enumerate(range(0, len(source), size)):
    (folder / f"paper.part{number:03d}").write_bytes(source[start:start + size])
print("Prepared paper parts for the Pages workflow.")
