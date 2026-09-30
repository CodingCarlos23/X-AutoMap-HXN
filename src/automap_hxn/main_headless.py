import sys
from pathlib import Path

from automap_hxn.workflows import mosaic_overlap_scan_auto_relative


def main():
    if len(sys.argv) < 2:
        print("Usage: automap-headless <path/to/config.json>")
        print("Example: automap-headless configs/simple_union_id-111111_t60_a60.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: config file not found: {json_path}")
        sys.exit(1)

    mosaic_overlap_scan_auto_relative(
        beamline_params=str(json_path),
        initial_scan_path=str(json_path),
    )

    print("\n[HEADLESS] All scans done.")


if __name__ == "__main__":
    main()
