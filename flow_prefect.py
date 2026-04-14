from prefect import flow, task
import subprocess
import sys 

@task
def run_pipeline():
    result = subprocess.run(
        [sys.executable, "pipeline.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

@flow(name="etl_ms_flow")
def etl_flow():
    run_pipeline()

if __name__ == "__main__":
    etl_flow()
