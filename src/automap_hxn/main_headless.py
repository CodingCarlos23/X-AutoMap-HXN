import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: automap-headless <path/to/config.json>")
        print("Example: automap-headless configs/simple_union_id-111111_t60_a60.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: config file not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        params = json.load(f)

    mp = params.get("mosaic_params", {})

    from automap_hxn.workflows import mosaic_overlap_scan_auto_relative

    mosaic_overlap_scan_auto_relative(
        beamline_params=str(json_path),
        initial_scan_path=str(json_path),
        xlen=mp.get("xlen", 100),
        ylen=mp.get("ylen", 100),
        overlap_per=mp.get("overlap_per", 5),
        dwell=mp.get("dwell", 0.01),
        step_size=mp.get("step_size", 0.25),
        mll=mp.get("mll", False),
        remote_seg=mp.get("remote_seg", False),
        followup_fine_scan=mp.get("followup_fine_scan", False),
        ref_scan_id=mp.get("ref_scan_id"),
    )

    print("\n[HEADLESS] All scans done.")


if __name__ == "__main__":
    main()
