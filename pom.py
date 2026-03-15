import re
if __name__ =="__main__":
    og_file = open("C:\Faks\Diplomski rad\data\GHA\examples\log\\1_Build ubuntu-latest wheels for x86_64.txt",'r',encoding='utf-8')
    tag_file  = open("1_Build ubuntu-latest wheels for x86_64.txt",'w',encoding='utf-8')

    line = og_file.readline()
    while line:
        line =  re.sub(r'/?([\w\-\.]+/)+',r'<path>',line)
        line = re.sub(r'([\w\-\.]+)\.(\s{1,6})',r' <filename>.\2',line)
        tag_file.write(line)
        line = og_file.readline()
    og_file.close()
    tag_file.close()
