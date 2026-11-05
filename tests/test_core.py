"""Input XYZ or Lab and print ICONS (ColorStimulus.u_name)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Input XYZ or Lab; output ICONS.")
    parser.add_argument("space", type=str.lower, choices=("xyz", "lab"))
    parser.add_argument("values", nargs=3, type=float, metavar="VALUE")
    parser.add_argument("--white", nargs=3, type=float, default=[95.047, 100, 108.883],
                        metavar="XYZ", help="Reference white XYZ; default D65.")
    argv = sys.argv[1:]
    if not argv:
        try:
            space = input("Colour space [xyz/lab, default xyz]: ").strip().lower() or "xyz"
            values = input("Enter three values separated by spaces or commas: ").replace(",", " ").replace("，", " ").split()
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
            return
        argv = [space, *values]
    args = parser.parse_args(argv)
    from colorstimuli import xyz_to_colorstimulus, lab_to_colorstimulus
    convert = xyz_to_colorstimulus if args.space == "xyz" else lab_to_colorstimulus
    try:
        print(convert(args.values, xyzw=args.white).u_name)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
