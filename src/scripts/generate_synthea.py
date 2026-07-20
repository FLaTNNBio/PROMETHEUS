from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def generate_synthea(
    synthea_dir: Path,
    output_dir: Path,
    population: int,
    seed: int,
    min_age: int,
    max_age: int,
    state: str,
) -> None:
    """Run Synthea on Windows and export the generated population as CSV."""

    runner = synthea_dir / "run_synthea.bat"

    if not runner.exists():
        raise FileNotFoundError(
            f"Synthea runner not found: {runner}. "
            "Clone Synthea and verify the supplied path."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # run_synthea.bat embeds command-line arguments in a Groovy expression.
    # Put paths and exporter settings in a temporary properties file so spaces
    # and Windows backslashes never reach that expression.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".properties",
        prefix="generate-synthea-",
        dir=synthea_dir,
        encoding="utf-8",
        delete=False,
    ) as config_file:
        config_file.write("exporter.csv.export = true\n")
        config_file.write("exporter.fhir.export = false\n")
        config_file.write(
            f"exporter.baseDirectory = {output_dir.resolve().as_posix()}\n"
        )
        config_path = Path(config_file.name)

    command = [
        str(runner),
        "-c",
        config_path.name,
        "-p",
        str(population),
        "-s",
        str(seed),
        "-a",
        f"{min_age}-{max_age}",
        state,
    ]

    print("Running command:")
    print(" ".join(command))

    try:
        subprocess.run(
            command,
            cwd=synthea_dir,
            check=True,
            shell=False,
        )
    finally:
        config_path.unlink(missing_ok=True)

    csv_dir = output_dir / "csv"

    if not csv_dir.exists():
        raise RuntimeError(
            f"Synthea completed but the CSV directory was not found: {csv_dir}"
        )

    print(f"Synthea data generated in: {csv_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic population using Synthea."
    )

    parser.add_argument(
        "--synthea-dir",
        type=Path,
        required=True,
        help="Path to the cloned Synthea repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthea"),
    )
    parser.add_argument("--population", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-age", type=int, default=50)
    parser.add_argument("--max-age", type=int, default=100)
    parser.add_argument(
        "--state",
        type=str,
        default="Massachusetts",
        help="Synthea geographic model to use.",
    )

    args = parser.parse_args()

    if args.population <= 0:
        parser.error("--population must be positive.")

    if args.min_age < 0 or args.max_age < args.min_age:
        parser.error("Invalid age interval.")

    generate_synthea(
        synthea_dir=args.synthea_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        population=args.population,
        seed=args.seed,
        min_age=args.min_age,
        max_age=args.max_age,
        state=args.state,
    )


if __name__ == "__main__":
    main()
