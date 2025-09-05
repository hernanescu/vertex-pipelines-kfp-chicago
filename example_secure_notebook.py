"""
Example of how to use the secure configuration in your notebooks
Replace the credential handling sections in your notebooks with this pattern
"""

# Instead of the old pattern:
# PROJECT_ID = ""
# if not os.getenv("IS_TESTING"):
#     shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
#     PROJECT_ID = shell_output[0]

# Use this new secure pattern:

from secure_config import setup_gcp_environment

# Initialize secure configuration
# This will automatically load .env file if present
config, client_factory = setup_gcp_environment()

# Access configuration values
PROJECT_ID = config.project_id
REGION = config.region
BUCKET_NAMES = config.bucket_names
BQ_CONFIG = config.bigquery_config

# Get authenticated clients
bq_client = client_factory.get_bigquery_client()
storage_client = client_factory.get_storage_client()

# Initialize AI Platform
client_factory.init_aiplatform()

# Use in your pipeline components
print(f"✅ Project: {PROJECT_ID}")
print(f"📍 Region: {REGION}")
print(f"🪣 Stage Bucket: {BUCKET_NAMES['stage_data']}")
print(f"📊 Historic Dataset: {BQ_CONFIG['historic_dataset']}")

# Example of using clients in your pipeline
def create_bigquery_dataset(dataset_name: str):
    """Example function using the secure client"""
    from google.cloud import bigquery
    
    dataset_id = f"{PROJECT_ID}.{dataset_name}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = BQ_CONFIG['location']
    
    try:
        dataset = bq_client.create_dataset(dataset, timeout=30)
        print(f"✅ Created dataset {dataset.dataset_id}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️  Dataset {dataset_id} already exists")
        else:
            raise e

# Example usage
if __name__ == "__main__":
    # This would replace the dataset creation logic in your setup notebook
    create_bigquery_dataset(BQ_CONFIG['historic_dataset'])
    create_bigquery_dataset(BQ_CONFIG['current_dataset'])