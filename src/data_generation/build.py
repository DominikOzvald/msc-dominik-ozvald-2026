import json
import random
import argparse

resistance = 1
mode = "standard"


def silent_anomaly():
    anomaly_choice = random.choice(["missing_build", "license"])
    if anomaly_choice == "missing_build":
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
    elif anomaly_choice == "license":
        package = random.choice(list(dependencies.keys()))
        print(f"<SILENT> WARN package {package} license requires action")
        license_status = random.choice(["expired", "near"])
        if license_status == "expired":
            print(f"<SILENT> WARN license expired {random.randint(1,30)} days ago")
            print(
                f"<SILENT> WARN license must be renewed to insure proper build operation"
            )
            print(
                f"<SILENT> WARN license can be renewed at https://{package}/licenses "
            )
        else:
            print(f"<SILENT> WARN license expires in {random.randint(1,30)}")
            print(f"<SILENT> No immediate risk to current build")
            print(
                f"<SILENT> WARN license can be renewed at https://{package}/licenses "
            )


def drift(version: str):
    prefix = ""
    new_version = version
    if version.startswith(("^", "~")):
        prefix = version[0]
        new_version = version[1:]

    change = random.choices(
        ["none", "major", "minor", "patch"], weights=[10 * resistance, 1, 1, 1], k=1
    )[0]
    new_version = [int(num) for num in new_version.split(".")]

    rand_int = random.randint(1, 3)

    if change == "major":
        new_version[0] += rand_int
        new_version[1] = 0
        new_version[2] = 0
    elif change == "minor":
        new_version[1] += rand_int
        new_version[2] = 0
    elif change == "patch":
        new_version[2] += rand_int

    new_version = [str(num) for num in new_version]
    new_version = f"{prefix}{'.'.join(new_version)}"

    return new_version, change


def dependency_check(dependencies):

    print(
        "INFO",
        "Initializing build dependency installation: Node v20.11.0, PNPM v8.15.4",
    )
    print("INFO", "Resolving dependency installation form package.json")
    patch_errors = 0
    minor_errors = 0
    major_errors = 0

    for dependency in dependencies:
        version = dependencies[dependency]
        new_version = dependencies[dependency]
        change = "none"
        if mode == "drift":
            new_version, change = drift(version)

        if change == "none":
            print(
                "INFO",
                f"Resolving dependency package {dependency} version f{new_version} matches config version {version}",
            )
        else:
            print("<DRIFT> WARN", "Dependency configuration has changed")
            print(
                "<DRIFT> WARN",
                f"Package {dependency} has new version {version} -> {new_version}",
            )

            if change == "patch":
                print("<DRIFT> WARN", "Patch update detected. No immediate risk")
                print("<DRIFT> WARN", "Preceding with package download anyway")
                patch_errors += 1
            elif change == "minor":
                print(
                    "<DRIFT> WARN",
                    f"Feature mismatch: New functionality in {dependency} might conflict with existing middleware.",
                )
                print("<DRIFT> WARN", "Build precedes with warnings")
                print("<DRIFT> WARN", "Preceding with package download anyway")
                minor_errors += 1

            elif change == "major":
                print(
                    "<DRIFT> ERROR",
                    "BREAKING CHANGE: Major version jump detected. API signatures may have changed.",
                )
                print(
                    "<DRIFT> ERROR",
                    "Build will fail after checking dependency configurations for other packages",
                )
                major_errors += 1

        if change in ["none", "minor", "patch"]:
            down = 0
            while down < 100:
                print("INFO", f"\tDownloading packages ... {down:.2f}%")
                down += random.randint(15, 35) + random.uniform(0, 1)
    print(
        "INFO",
        f"added {random.randint(1,15)} packages, audited 536 packages in {random.randint(5,30)}s",
    )

    print("INFO", f"{random.randint(25,70)} package is looking for funding")
    print("INFO", f"run `npm fund` for details")

    if major_errors or minor_errors or patch_errors:
        print(
            "<DRIFT> WARN",
            f"{major_errors+minor_errors+patch_errors} vulnerabilities ({patch_errors} moderate, {minor_errors} high, {major_errors} critical)",
        )
    else:
        print("INFO", f"found 0 vulnerabilities")

    if major_errors:
        print(
            "<DRIFT> ERROR",
            "Exiting build process due to major difference in package version",
        )
        print("<DRIFT> INFO", "Run `npm audit` for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="standard", required=False, type=str)
    parser.add_argument("-r", "--resistance", default=1, required=False, type=float)

    args = parser.parse_args()
    mode = args.mode
    resistance = args.resistance

    with open("package.json", "r") as file:
        package = json.load(file)
        dependencies = package["dependencies"]

    print("\n\nINFO", "> build")
    print("INFO", "> npm run clean && tsc && npm run copy-views && npm run copy-public")
    print("\n\nINFO", "> clean")
    print("INFO", "> shx rm -rf dist")

    dependency_check(dependencies)
    if mode == "silent":
        silent_anomaly()

    print("\n\nINFO", "> copy-views")
    print("INFO", "> shx cp -r src/views dist")
    print("\n\nINFO", "> copy-public")
    print("INFO", "> shx cp -r src/public dist/public")
