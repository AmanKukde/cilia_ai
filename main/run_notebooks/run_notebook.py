import argparse
import os
from datetime import datetime
import papermill as pm
import random
def main():
    parser = argparse.ArgumentParser(description="Run a notebook with Papermill")

    parser.add_argument(
        "--notebook", type=str, required=True,
        help="Path to the input .ipynb notebook"
    )
    parser.add_argument(
        "--outputdir", type=str, required=True,
        help="Directory to save executed notebook"
    )
    parser.add_argument(
        "-p", "--param", action="append", default=[],
        help="Notebook parameters as key=value pairs"
    )

    args = parser.parse_args()

    # Build parameter dict
    parameters = {}
    for p in args.param:
        if "=" not in p:
            raise ValueError(f"Invalid parameter format: {p}. Use key=value.")
        key, val = p.split("=", 1)
        # Try ints or floats automatically
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass
        parameters[key] = val

    # Timestamp and output path
    timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    os.makedirs(args.outputdir, exist_ok=True)

    notebook_name = os.path.basename(args.notebook).replace(".ipynb", "")
    output_path = os.path.join(args.outputdir, f"{timestamp}_{notebook_name}_{random.randint(1000, 9999) }.ipynb")

    print("\n📘 Running notebook")
    print(f"  Input:   {args.notebook}")
    print(f"  Output:  {output_path}")
    print(f"  Params:  {parameters}")
    print("==========================================")

    # Execute the notebook
    pm.execute_notebook(
        input_path=args.notebook,
        output_path=output_path,
        parameters=parameters,
        log_output=True
    )

    print("\n✅ Notebook execution complete.")
    print(f"Saved: {output_path}\n")

if __name__ == "__main__":
    main()
