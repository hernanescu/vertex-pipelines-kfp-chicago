"""
Secure configuration management for GCP ML Pipeline
Provides centralized, secure credential and configuration management
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery, aiplatform, storage
import google.auth.transport.requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCPConfig:
    """Centralized GCP configuration management with secure credential handling"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize GCP configuration
        
        Args:
            env_file: Path to .env file (optional)
        """
        if env_file:
            self._load_env_file(env_file)
            
        self.project_id = self._get_project_id()
        self.region = os.getenv('GCP_REGION', 'us-central1')
        self.credentials, self.auth_project = self._get_credentials()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"✅ GCP Config initialized for project: {self.project_id}")
        logger.info(f"📍 Region: {self.region}")
    
    def _load_env_file(self, env_file: str):
        """Load environment variables from file"""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"📁 Loaded environment from: {env_file}")
        except ImportError:
            logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
        except Exception as e:
            logger.warning(f"Could not load env file {env_file}: {e}")
    
    def _get_project_id(self) -> str:
        """Get project ID from multiple sources with fallback"""
        # Priority: Environment variable > gcloud config > metadata service
        
        # 1. Environment variable (highest priority)
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if project_id:
            logger.info("🔧 Project ID from environment variable")
            return project_id
        
        # 2. gcloud config
        try:
            import subprocess
            result = subprocess.run(
                ['gcloud', 'config', 'get-value', 'project'], 
                capture_output=True, text=True, check=True, timeout=10
            )
            project_id = result.stdout.strip()
            if project_id and project_id != "(unset)":
                logger.info("⚙️  Project ID from gcloud config")
                return project_id
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"Could not get project from gcloud: {e}")
        
        # 3. Metadata service (works on GCP compute instances)
        try:
            import requests
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/project/project-id',
                headers={'Metadata-Flavor': 'Google'},
                timeout=5
            )
            if response.status_code == 200:
                logger.info("☁️  Project ID from metadata service")
                return response.text
        except Exception as e:
            logger.debug(f"Could not get project from metadata service: {e}")
        
        raise ValueError(
            "❌ Could not determine GCP project ID. Please set GOOGLE_CLOUD_PROJECT "
            "environment variable or configure gcloud CLI"
        )
    
    def _get_credentials(self):
        """Get default credentials with error handling"""
        try:
            credentials, project = default()
            
            # Refresh credentials to validate them
            request = google.auth.transport.requests.Request()
            credentials.refresh(request)
            
            logger.info("🔐 Using default credentials")
            return credentials, project or self.project_id
            
        except DefaultCredentialsError as e:
            raise ValueError(
                f"❌ No valid GCP credentials found: {e}\n"
                "Please run 'gcloud auth application-default login' or set "
                "GOOGLE_APPLICATION_CREDENTIALS environment variable"
            )
    
    def _validate_config(self):
        """Validate configuration values"""
        if not self.project_id:
            raise ValueError("Project ID cannot be empty")
        
        if not self.region:
            raise ValueError("Region cannot be empty")
        
        # Validate region format
        valid_regions = ['us-central1', 'us-east1', 'us-west1', 'europe-west1', 'asia-southeast1']
        if self.region not in valid_regions:
            logger.warning(f"⚠️  Region '{self.region}' may not be valid. Common regions: {valid_regions}")
    
    @property
    def bucket_names(self) -> Dict[str, str]:
        """Get standardized bucket names"""
        return {
            'stage_data': os.getenv('STAGE_DATA_BUCKET', f'{self.project_id}-chicago-taxi-stage'),
            'pipeline_data': os.getenv('PIPELINE_BUCKET', f'{self.project_id}-chicago-taxi-pipelines'),
        }
    
    @property
    def bigquery_config(self) -> Dict[str, str]:
        """Get BigQuery configuration"""
        return {
            'historic_dataset': os.getenv('BQ_DATASET_HISTORIC', 'chicago_taxi_historic'),
            'current_dataset': os.getenv('BQ_DATASET_CURRENT', 'chicago_taxi_current'),
            'location': os.getenv('BQ_LOCATION', 'US'),
        }


class GCPClientFactory:
    """Factory for creating authenticated GCP clients"""
    
    def __init__(self, config: GCPConfig):
        self.config = config
        self._clients = {}
        logger.info("🏭 GCP Client Factory initialized")
    
    def get_bigquery_client(self) -> bigquery.Client:
        """Get or create BigQuery client"""
        if 'bigquery' not in self._clients:
            self._clients['bigquery'] = bigquery.Client(
                project=self.config.project_id,
                credentials=self.config.credentials,
                location=self.config.bigquery_config['location']
            )
            logger.info("📊 BigQuery client created")
        return self._clients['bigquery']
    
    def get_storage_client(self) -> storage.Client:
        """Get or create Storage client"""
        if 'storage' not in self._clients:
            self._clients['storage'] = storage.Client(
                project=self.config.project_id,
                credentials=self.config.credentials
            )
            logger.info("🪣 Storage client created")
        return self._clients['storage']
    
    def init_aiplatform(self):
        """Initialize AI Platform with proper credentials"""
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            credentials=self.config.credentials
        )
        logger.info("🤖 AI Platform initialized")
    
    def validate_access(self) -> Dict[str, str]:
        """Validate access to all required GCP services"""
        results = {}
        
        # Test BigQuery access
        try:
            bq_client = self.get_bigquery_client()
            list(bq_client.list_datasets(max_results=1))
            results['bigquery'] = '✅ Access granted'
        except Exception as e:
            results['bigquery'] = f'❌ Access denied: {str(e)}'
        
        # Test AI Platform access
        try:
            self.init_aiplatform()
            results['aiplatform'] = '✅ Access granted'
        except Exception as e:
            results['aiplatform'] = f'❌ Access denied: {str(e)}'
        
        # Test Storage access
        try:
            storage_client = self.get_storage_client()
            list(storage_client.list_buckets(max_results=1))
            results['storage'] = '✅ Access granted'
        except Exception as e:
            results['storage'] = f'❌ Access denied: {str(e)}'
        
        return results


def create_env_template(output_path: str = ".env.template"):
    """Create a template .env file"""
    template_content = """# GCP ML Pipeline Configuration
# Copy this file to .env and fill in your values

# Required: Your GCP Project ID
GOOGLE_CLOUD_PROJECT=your-project-id-here

# Optional: Override default region
GCP_REGION=us-central1

# Optional: Override default bucket names
STAGE_DATA_BUCKET=your-project-stage-bucket
PIPELINE_BUCKET=your-project-pipeline-bucket

# Optional: Override BigQuery configuration
BQ_DATASET_HISTORIC=chicago_taxi_historic
BQ_DATASET_CURRENT=chicago_taxi_current
BQ_LOCATION=US

# Optional: Explicit service account key path (NOT recommended for production)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Pipeline-specific settings
RF_HP_IMAGE=gcr.io/your-project/rf_hp_job:v1
LR_HP_IMAGE=gcr.io/your-project/lr_hp_job:v1
"""
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    logger.info(f"📝 Created environment template: {output_path}")
    logger.info("👉 Copy this file to .env and update with your values")


# Convenience function for quick setup
def setup_gcp_environment(env_file: Optional[str] = None) -> tuple[GCPConfig, GCPClientFactory]:
    """
    Quick setup function for GCP environment
    
    Args:
        env_file: Path to .env file (optional)
        
    Returns:
        Tuple of (config, client_factory)
    """
    config = GCPConfig(env_file)
    client_factory = GCPClientFactory(config)
    
    # Validate access
    logger.info("🔍 Validating GCP service access...")
    access_results = client_factory.validate_access()
    
    for service, status in access_results.items():
        logger.info(f"  {service}: {status}")
    
    return config, client_factory


if __name__ == "__main__":
    # Create template file when run directly
    create_env_template()
    print("\n🚀 To get started:")
    print("1. Copy .env.template to .env")
    print("2. Fill in your GCP project ID and other values")
    print("3. Import and use: from secure_config import setup_gcp_environment")