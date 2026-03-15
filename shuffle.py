import random

f = open("C:\Faks\Diplomski rad\code\\test_data\\6_Test 3.9 x64 wheels for windows-latest.txt","r",encoding="utf-8")
lines = f.readlines()
f.close()
random.shuffle(lines)
f = open("C:\Faks\Diplomski rad\code\\test_data\\6_Test 3.9 x64 wheels for windows-latest_rand.txt","w",encoding="utf-8")
f.writelines(lines)
f.close()