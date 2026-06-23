import unittest
import json
import argparse
import random
import sys

data = {}
mode = "standard"
server_address = "127.0.0.1"
resistance = 1

db_user = random.choice(
    ["root", "postgres", "admin", "user", "aegilops", "truffelhog", "lionbrave"]
)
db_host = f"https://{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
db_name = random.choice(
    ["db_apuw_subjects", "db_subjects", "db", "geeksforgeeks", "storage_db", "dummy_db"]
)
user = random.choice(["aegilops", "truffelhog", "lionbrave"])
secrets = {
    "db": {
        "db_user": db_user,
        "host": db_host,
        "password": random.randbytes(16).hex(),
        "db_name": db_name,
        "db_port": random.randint(1000, 9999),
    },
    "user": {
        "username": user,
        "password": random.randbytes(16).hex(),
        "role": random.choice(["student", "teacher", "admin"]),
    },
    "oAuth2": {
        "client_id": random.randbytes(16).hex(),
        "client_secret": random.randbytes(16).hex(),
        "secret": random.randbytes(16).hex(),
        "signing_secret": random.randbytes(16).hex(),
        "signing_alg": "HS256",
    },
}


def secret_anomaly():
    leak_choice = random.choices(
        ["none", "user", "oAuth2", "db"], weights=[4 * resistance, 1, 1, 1], k=1
    )[0]
    if leak_choice in secrets:
        leak = secrets[leak_choice]
        print()
        print(f"<SECURITY> [debug] {leak_choice}", "{")
        for key in leak:
            print(f"<SECURITY> {key} : {leak[key]}")
        print("<SECURITY> }")


def debug_print():
    value = random.choices(
        [
            "adminUser",
            "regularUser",
            "subjects",
            "exams",
            "attendance",
            "updatedSubject",
            "updatedExam",
            "none",
        ],
        weights=[1,1,1,1,1,1,1,10],
        k=1
    )[0]
    if value in data:
        print(f"\n[debug] {value} :")
        try:
            print(json.dumps(data[value]["rows"][0], indent=4))
        except:
            print(json.dumps(data[value], indent=4))


def silent_anomaly(method: str):

    anomaly_choice = random.choices(
        ["no_test_value_file", "no_test_values", "missing_build"],
        weights=[1, 1, 1],
        k=1,
    )[0]

    if anomaly_choice != "none":
        print(f"<SILENT> WARN Testing Method {method} terminated")

        if anomaly_choice == "no_test_value_file":
            test_file = random.choice(
                [
                    "pom/test_value",
                    "tmp/test_value",
                    "pom/test_values",
                    "pom/test_values",
                    "values",
                    "tests",
                    "tests/values",
                    "pom",
                ]
            )
            print(
                f"<SILENT> ERROR test.py :{random.randint(50,75)}:{len(test_file)}: fatal error occurred: {test_file}.json no such file or directory"
            )
            print(f"<SILENT> ERROR with open('{test_file}.json', 'r') as file:")
            print(f"<SILENT> ERROR           ^{'~'*(len(test_file)+5)}")

        elif anomaly_choice == "no_test_values":
            values = random.choices(
                [
                    "adminUser",
                    "regularUser",
                    "subjects",
                    "exams",
                    "attendance",
                    "updatedSubject",
                    "updatedExam",
                ],
                k=random.randint(1, 7),
            )
            print(f"<SILENT> Test values resolved successfully")
            print(f"<SILENT> Missing required test values for method {method}")
            print(f"<SILENT> Required values are {values}")
        elif anomaly_choice == "missing_build":
            n = random.randint(3, 6)
            directories = [
                "pom",
                "build",
                "dist",
                "public",
                "docs",
                "routes",
                "utils",
                "views",
                "public/js",
            ]
            selected_dirs = random.choices(directories, k=n)
            extensions = [".inc", ".conf", ".*_t", ".js", ".css", ".h5"]
            random.shuffle(extensions)
            selected_extensions = extensions[:n]
            for extension, directory in zip(selected_extensions, selected_dirs):
                print(
                    f"<SILENT> WARN no files found matching '*{extension}' under directory {directory}"
                )

        print("ok")


class TestGetMethods(unittest.TestCase):

    def setUp(self):
        self.urls = [
            "/api/users",
            "/api/subjects",
            "/api/subjects/{idSubject}",
            "/api/subjects/{idSubject}/students",
            "/api/subjects/{idSubject}/students/{idStudent}/exams",
            "/api/subjects/{idSubject}/students/{idStudent}/exams/{idExam}",
        ]
        self.bodies_expected = [
            data["regularUser"]["rows"],
            data["subjects"]["rows"],
            data["subjects"]["rows"][0],
            data["regularUser"]["rows"],
            data["exams"]["rows"],
            data["exams"]["rows"][0],
        ]
        self.bodies_received = self.bodies_expected.copy()

    def test_found(self):
        if mode == "security":
            secret_anomaly()
        if mode != "flaky":
            debug_print()
        for url, body_expected, body_received in zip(
            self.urls, self.bodies_expected, self.bodies_received
        ):
            with self.subTest(url=url):
                status = 200
                if mode == "flaky":
                    status, body_received = random.choices(
                        [
                            (200, body_expected),
                            (404, f"404 Can not find {url}"),
                            (500, "Server error"),
                        ],
                        weights=[10 * resistance, 1, 1],
                        k=1,
                    )[0]
                self.assertEqual((status, body_received), (200, body_expected))

    def test_authorization(self):
        if mode == "security":
            secret_anomaly()
        if mode != "flaky":
            debug_print()
        for url, body_expected, body_received in zip(
            self.urls, self.bodies_expected, self.bodies_received
        ):
            with self.subTest(url=url):
                status = 200
                if mode == "flaky":
                    status, body_received = random.choices(
                        [
                            (200, body_expected),
                            (403, f"403 You do not have permission to GET {url}"),
                            (500, "Server error"),
                        ],
                        weights=[10 * resistance, 1, 1],
                        k=1,
                    )[0]
                self.assertEqual((status, body_received), (200, body_expected))


class TestPostMethods(unittest.TestCase):

    def setUp(self):
        self.urls = [
            "/api/subjects",
            "/api/subjects/{idSubject}/students/{idStudent}/exams",
        ]
        self.locations_expected = [
            f"{server_address}/api/subjects/0",
            f"{server_address}/api/subjects/0/students/0/exams/0",
        ]
        self.locations_received = self.locations_expected.copy()

    def test_created(self):
        if mode == "security":
            secret_anomaly()
        if mode != "flaky":
            debug_print()
        for url, location_expected, location_received in zip(
            self.urls, self.locations_expected, self.locations_received
        ):
            with self.subTest(url=url):
                headers = {"status": 201, "location": location_received}
                headers_expected = {"status": 201, "location": location_expected}
                if mode == "flaky":
                    headers = random.choices(
                        [
                            {"status": 201, "location": location_received},
                            {"status": 401, "location": ""},
                            {"status": 403, "location": ""},
                            {"status": 500, "location": ""},
                        ],
                        weights=[10 * resistance, 1, 1, 1],
                        k=1,
                    )[0]

                self.assertEqual(headers, headers_expected)


class TestPutMethods(unittest.TestCase):
    def setUp(self):
        self.urls = [
            "/api/subjects/{idSubject}",
            "/api/subjects/{idSubject}/students/{idStudent}/exams",
            "/api/subjects/{idSubject}/students/{idStudent}/exams/{idExam}",
        ]
        self.responses = [
            {"status": 200, "text": "Update successful"},
            {
                "status": 201,
                "location": f"{server_address}/api/subjects/0/students/0/exams",
            },
            {"status": 200, "text": "Update successful"},
        ]

    def test_update(self):
        if mode == "security":
            secret_anomaly()
        if mode != "flaky":
            debug_print()
        for url, response_expected in zip(self.urls, self.responses):
            with self.subTest(url=url):
                response_received = response_expected
                if mode == "flaky":
                    response_received = random.choices(
                        [
                            response_expected,
                            {"status": 200, "text": "Update successful"},
                            {
                                "status": 201,
                                "location": f"{server_address}/api/subjects/0/students/0/exams",
                            },
                            {"status": 401, "text": "Authentication is needed"},
                            {
                                "status": 403,
                                "text": "403 You do not have permission to PUT {url}",
                            },
                            {"status": 500, "text": "Server error"},
                        ],
                        weights=[10 * resistance, 1, 1, 1, 1, 1],
                        k=1,
                    )[0]
                self.assertEqual(response_received, response_expected)


class TestDeleteMethods(unittest.TestCase):

    def setUp(self):
        self.urls = [
            "/api/subjects/{idSubject}",
            "/api/subjects/{idSubject}/students/{idStudent}/exams",
            "/api/subjects/{idSubject}/students/{idStudent}/exams/{idExam}",
        ]

    def test_delete(self):
        if mode == "security":
            secret_anomaly()
        if mode != "flaky":
            debug_print()
        for url in self.urls:
            with self.subTest(url=url):
                response_expected = {"status": 204}
                response_received = {"status": 204}
                if mode == "flaky":
                    response_received = random.choices(
                        [
                            response_expected,
                            {"status": 200, "text": "deletion successful"},
                            {"status": 500, "text": "server error"},
                            {"status": 404, "text": "not found"},
                            {
                                "status": 403,
                                "text": "403 You do not have permission to DELETE {url}",
                            },
                            {"status": 401, "text": "Authentication is needed"},
                        ],
                        weights=[10 * resistance, 1, 1, 1, 1, 1],
                        k=1,
                    )[0]
                self.assertEqual(response_received, response_expected)


if __name__ == "__main__":
    with open("test_values.json", "r") as file:
        data = json.load(file)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="standard", required=False, type=str)
    parser.add_argument("-r", "--resistance", default=1, required=False, type=float)

    args = parser.parse_args()
    mode = args.mode
    resistance = args.resistance

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    loader = unittest.TestLoader()
    num_eq = 30

    for method, test_case in zip(
        ["GET", "POST", "PUT", "DELETE"],
        [TestGetMethods, TestPostMethods, TestPutMethods, TestDeleteMethods],
    ):
        print(f"\n{'='*num_eq}Testing {method} Methods{'='*num_eq}")
        allow_silent_anomaly = random.choices(
            [False, True], weights=[4 * resistance, 1], k=1
        )[0]
        if mode == "silent" and allow_silent_anomaly:
            silent_anomaly(method)

        else:

            result = runner.run(loader.loadTestsFromTestCase(test_case))
