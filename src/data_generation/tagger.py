import sys


def tag_flaky(lines):
    for i in range(len(lines)):
        if len(lines[i]) > 3 and not (
            lines[i].startswith("--------------")
            or lines[i].startswith("==================")
            or lines[i].startswith("         ")
        ):
            lines[i] = f"<FLAKY> " + lines[i]


if __name__ == "__main__":
    inputs = []
    for line in sys.stdin:

        tmp = line.split("\n")
        for part in tmp[:-1]:
            inputs.append(part)

        if line.startswith("OK"):
            print("\n".join(inputs))
            inputs = []
        elif line.startswith("FAILED"):
            tag_flaky(inputs)
            print("\n".join(inputs))
            inputs = []
