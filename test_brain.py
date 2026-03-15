from Brain import LogParser
if __name__ =="__main__":

    in_dir = "C:\Faks\Diplomski rad\data\GHA\examples\log"
    out_dir = "./out_brain"
    log_file = "1_Build ubuntu-latest wheels for x86_64.txt"
    log_format = "<DateTime> <Content>"
    st=0.2
    depth=6
    child_limit=40
    re = [r'##\[\w+\]\.*']
    delimiter = []
    threshold = 2
    # files = [file for file in os.listdir(in_dir) if file[-4:] ==".txt"]
    # print(files)




    parser = LogParser("logs",log_format,in_dir,out_dir, threshold= threshold,rex = re,delimeter=delimiter)
    parser.parse("6_Test 3.9 x64 wheels for windows-latest.txt")
    parser.parse("7_Test 3.10 x64 wheels for windows-latest.txt")