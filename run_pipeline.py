import click
import logging
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    logger.info("🚀 Starting the ML training pipeline...")

    # Run the ZenML pipeline
    pipeline_instance = ml_pipeline()
    run = pipeline_instance.run()   # <-- explicitly call .run() to execute

    logger.info("✅ Pipeline finished successfully.")

    print("\n To inspect experiment runs in MLflow, run the following in your terminal:\n")
    print(f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n")
    print("Then open http://127.0.0.1:5000 in your browser.\n")
    print("You’ll find all experiment runs tracked under the experiment: 'flight_price_predictor'.")


if __name__ == "__main__":
    main()
