import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from massscriber import profiles


class ProfileTests(TestCase):
    def test_save_load_list_and_delete_profiles(self):
        with TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / ".massscriber"
            profile_file = state_dir / "profiles.json"
            with patch.object(profiles, "APP_STATE_DIR", state_dir), patch.object(
                profiles,
                "PROFILE_FILE",
                profile_file,
            ):
                profiles.save_profile("Demo", {"model": "turbo", "watch_folder_path": "C:/incoming"})

                self.assertEqual(profiles.list_profile_names(), ["Demo"])
                self.assertEqual(profiles.get_profile("Demo")["model"], "turbo")

                profiles.save_profile("Podcast", {"model": "large-v3"})
                self.assertEqual(profiles.list_profile_names(), ["Demo", "Podcast"])

                payload = json.loads(profile_file.read_text(encoding="utf-8"))
                self.assertIn("Demo", payload)

                profiles.delete_profile("Demo")
                self.assertEqual(profiles.list_profile_names(), ["Podcast"])
