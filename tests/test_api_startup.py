from unittest.mock import Mock, sentinel

from fastapi.testclient import TestClient

from models.configs import MODEL_REGISTRY
from serve import api


def test_default_variant_through_app_lifespan(monkeypatch):
    empty_cache = {"variant_key": None, "model": None, "tokenizer": None}

    def load_registered_variant(variant_key):
        if variant_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown variant '{variant_key}'")
        return sentinel.model, sentinel.tokenizer, {}

    load_model = Mock(side_effect=load_registered_variant)
    unload_model = Mock()
    monkeypatch.setattr(api, "_cache", empty_cache.copy())
    monkeypatch.setattr(api, "load_model", load_model)
    monkeypatch.setattr(api, "unload_model", unload_model)

    with TestClient(api.app) as client:
        load_model.assert_called_once_with("llama_nf4")
        unload_model.assert_not_called()
        assert api._cache["model"] is sentinel.model
        assert api._cache["tokenizer"] is sentinel.tokenizer

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model": "llama_nf4"}

        assert api.GenerateRequest(prompt="Hello").variant == "llama_nf4"

    unload_model.assert_called_once_with(sentinel.model)
    assert api._cache == empty_cache
