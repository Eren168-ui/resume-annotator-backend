import asyncio
import os
import json
import unittest
from unittest import mock

from api.auth import login
from models.schemas import LoginRequest


class AuthConfigTests(unittest.TestCase):
    def test_login_reads_users_from_auth_users_json(self):
        users = [
            {
                "id": "u1",
                "email": "snapshot-admin@example.com",
                "name": "管理员 1",
                "password": "snapshot-pass",
            },
            {
                "id": "u2",
                "email": "snapshot-reviewer@example.com",
                "name": "管理员 2",
                "password": "another-pass",
            },
        ]
        with mock.patch.dict(
            os.environ,
            {
                "AUTH_USERS_JSON": json.dumps(users, ensure_ascii=False),
                "DEV_AUTH_BYPASS": "false",
            },
            clear=False,
        ):
            response = asyncio.run(
                login(LoginRequest(email="snapshot-admin@example.com", password="snapshot-pass"))
            )

        self.assertIn("token", response)
        self.assertEqual(response["user"]["email"], "snapshot-admin@example.com")
        self.assertEqual(response["user"]["name"], "管理员 1")

    def test_login_rejects_invalid_password_when_bypass_disabled(self):
        users = [
            {
                "id": "u1",
                "email": "snapshot-admin@example.com",
                "name": "管理员 1",
                "password": "snapshot-pass",
            }
        ]
        with mock.patch.dict(
            os.environ,
            {
                "AUTH_USERS_JSON": json.dumps(users, ensure_ascii=False),
                "DEV_AUTH_BYPASS": "false",
            },
            clear=False,
        ):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(
                    login(LoginRequest(email="snapshot-admin@example.com", password="wrong-pass"))
                )

        self.assertEqual(getattr(ctx.exception, "status_code", None), 401)


if __name__ == "__main__":
    unittest.main()
