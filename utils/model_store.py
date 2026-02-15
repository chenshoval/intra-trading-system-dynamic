"""Model artifact storage — Azure Blob + ACR.

Handles saving/loading trained models to:
- Local filesystem (development)
- Azure Blob Storage (production)
- Azure Container Registry (for containerized inference)
"""

import os
import io
import json
import pickle
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("trading.model_store")


class ModelStore:
    """Manage model artifacts across local and Azure storage.

    Usage:
        store = ModelStore()  # local only
        store = ModelStore(azure_container="models")  # Azure Blob
        store.save(model, "lightgbm_v1", metadata={...})
        model = store.load("lightgbm_v1")
    """

    def __init__(
        self,
        local_dir: str = "models",
        azure_container: Optional[str] = None,
        azure_account_url: Optional[str] = None,
    ):
        self.local_dir = local_dir
        self.azure_container = azure_container
        self.azure_account_url = azure_account_url or os.environ.get(
            "ACCOUNT_URL", "https://tradingsystemsa.blob.core.windows.net"
        )
        os.makedirs(local_dir, exist_ok=True)

    def save(
        self,
        model: Any,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[dict] = None,
        upload_to_azure: bool = False,
    ) -> str:
        """Save model locally and optionally to Azure Blob.

        Args:
            model: The model object (must be picklable, or LightGBM/XGBoost)
            model_name: Name identifier
            version: Version string (default: timestamp)
            metadata: Dict of metadata to save alongside
            upload_to_azure: Also upload to Azure Blob Storage

        Returns:
            Path where model was saved
        """
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        model_dir = os.path.join(self.local_dir, model_name, version)
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        self._save_model_file(model, model_path)

        # Save metadata
        meta = {
            "model_name": model_name,
            "version": version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_type": type(model).__name__,
            **(metadata or {}),
        }
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Saved model: {model_dir}")

        # Upload to Azure if requested
        if upload_to_azure and self.azure_container:
            self._upload_to_azure(model_dir, model_name, version)

        return model_dir

    def load(
        self,
        model_name: str,
        version: Optional[str] = None,
        from_azure: bool = False,
    ) -> Any:
        """Load a model from local storage or Azure.

        Args:
            model_name: Name identifier
            version: Specific version (default: latest)
            from_azure: Download from Azure first

        Returns:
            The loaded model object
        """
        if from_azure and self.azure_container:
            self._download_from_azure(model_name, version)

        model_dir = os.path.join(self.local_dir, model_name)

        if version is None:
            # Find latest version
            versions = sorted(os.listdir(model_dir)) if os.path.exists(model_dir) else []
            if not versions:
                raise FileNotFoundError(f"No versions found for {model_name}")
            version = versions[-1]

        model_path = os.path.join(model_dir, version, "model.pkl")
        model = self._load_model_file(model_path)
        logger.info(f"Loaded model: {model_name}/{version}")
        return model

    def get_metadata(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> dict:
        """Get metadata for a saved model."""
        model_dir = os.path.join(self.local_dir, model_name)

        if version is None:
            versions = sorted(os.listdir(model_dir)) if os.path.exists(model_dir) else []
            if not versions:
                raise FileNotFoundError(f"No versions found for {model_name}")
            version = versions[-1]

        meta_path = os.path.join(model_dir, version, "metadata.json")
        with open(meta_path) as f:
            return json.load(f)

    def list_models(self) -> list[dict]:
        """List all saved models and their versions."""
        models = []
        if not os.path.exists(self.local_dir):
            return models

        for model_name in sorted(os.listdir(self.local_dir)):
            model_dir = os.path.join(self.local_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
            versions = sorted(os.listdir(model_dir))
            for v in versions:
                try:
                    meta = self.get_metadata(model_name, v)
                    models.append(meta)
                except Exception:
                    models.append({"model_name": model_name, "version": v})
        return models

    # ── Private helpers ──

    def _save_model_file(self, model: Any, path: str) -> None:
        """Save model using the best available method."""
        model_type = type(model).__name__

        if model_type in ("LGBMClassifier", "LGBMRegressor", "Booster"):
            # LightGBM native save
            model.save_model(path.replace(".pkl", ".lgbm"))
            # Also save pickle for generic loading
            with open(path, "wb") as f:
                pickle.dump(model, f)
        elif model_type in ("XGBClassifier", "XGBRegressor"):
            # XGBoost native save
            model.save_model(path.replace(".pkl", ".xgb"))
            with open(path, "wb") as f:
                pickle.dump(model, f)
        else:
            # Generic pickle
            with open(path, "wb") as f:
                pickle.dump(model, f)

    def _load_model_file(self, path: str) -> Any:
        """Load model from pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _upload_to_azure(self, local_dir: str, model_name: str, version: str) -> None:
        """Upload model directory to Azure Blob."""
        from utils.data_loaders import upload_to_blob

        for filename in os.listdir(local_dir):
            filepath = os.path.join(local_dir, filename)
            if not os.path.isfile(filepath):
                continue

            blob_path = f"models/{model_name}/{version}/{filename}"
            with open(filepath, "rb") as f:
                upload_to_blob(
                    f.read(),
                    blob_path,
                    container=self.azure_container,
                    account_url=self.azure_account_url,
                )

        logger.info(f"Uploaded to Azure: models/{model_name}/{version}/")

    def _download_from_azure(self, model_name: str, version: Optional[str] = None) -> None:
        """Download model from Azure Blob to local."""
        from utils.data_loaders import download_from_blob, list_blobs

        prefix = f"models/{model_name}/"
        if version:
            prefix += f"{version}/"

        blobs = list_blobs(prefix=prefix, container=self.azure_container,
                          account_url=self.azure_account_url)

        for blob_name in blobs:
            content = download_from_blob(
                blob_name,
                container=self.azure_container,
                account_url=self.azure_account_url,
                as_dataframe=False,
            )
            local_path = os.path.join(self.local_dir, blob_name.replace("models/", ""))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(content)

        logger.info(f"Downloaded from Azure: {prefix}")
