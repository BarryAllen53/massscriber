import os
from unittest import TestCase
from unittest.mock import patch

from massscriber.providers import (
    get_provider_api_key,
    get_provider_default_model,
    get_provider_models,
    normalize_provider_name,
    provider_file_limit_warning,
    provider_supports_translation,
)


class ProviderTests(TestCase):
    def test_normalize_provider_name_defaults_to_local(self):
        self.assertEqual(normalize_provider_name(None), "local")
        self.assertEqual(normalize_provider_name("unknown"), "local")
        self.assertEqual(normalize_provider_name("Groq"), "groq")

    def test_get_provider_api_key_uses_env_fallback(self):
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "dg-secret"}, clear=False):
            self.assertEqual(get_provider_api_key("deepgram", ""), "dg-secret")

    def test_provider_models_and_translation_flags(self):
        self.assertIn("whisper-1", get_provider_models("openai"))
        self.assertEqual(get_provider_default_model("groq"), "whisper-large-v3-turbo")
        self.assertTrue(provider_supports_translation("openai"))
        self.assertFalse(provider_supports_translation("assemblyai"))

    def test_provider_file_limit_warning_flags_large_uploads(self):
        warning = provider_file_limit_warning("openai", 30 * 1024 * 1024)
        self.assertIsNotNone(warning)
        self.assertIn("25 MB", warning or "")
