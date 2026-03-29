import asyncio
import os
import unittest
from unittest import mock

from api.auth import login
from models.schemas import LoginRequest


class AuthConfigTests(unittest.TestCase):
    def test_login_reads_admin_credentials_from_current_env(self):
        with mock.patch.dict(
            os.environ,
            {
                "ADMIN_EMAIL": "snapshot-admin@example.com",
                "ADMIN_PASSWORD": "snapshot-pass",
            },
            clear=False,
        ):
            response = asyncio.run(
                login(LoginRequest(email="snapshot-admin@example.com", password="snapshot-pass"))
            )

        self.assertIn("token", response)
        self.assertEqual(response["user"]["email"], "snapshot-admin@example.com")


if __name__ == "__main__":
    unittest.main()
