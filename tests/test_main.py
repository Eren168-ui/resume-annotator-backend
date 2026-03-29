import unittest

import main


class AllowedOriginsTests(unittest.TestCase):
    def test_build_allowed_origins_supports_csv_and_dedupes(self):
        origins = main.build_allowed_origins(
            frontend_origin="https://erenlab.cn",
            frontend_origins="https://www.erenlab.cn, https://erenlab.cn , https://app.erenlab.cn",
            local_frontend_origins=["http://localhost:5173"],
        )
        self.assertEqual(
            origins,
            [
                "https://erenlab.cn",
                "https://www.erenlab.cn",
                "https://app.erenlab.cn",
                "http://localhost:5173",
            ],
        )

    def test_build_allowed_origins_ignores_empty_values(self):
        origins = main.build_allowed_origins(
            frontend_origin="",
            frontend_origins=" ,  , ",
            local_frontend_origins=["http://localhost:4173"],
        )
        self.assertEqual(origins, ["http://localhost:4173"])


if __name__ == "__main__":
    unittest.main()
