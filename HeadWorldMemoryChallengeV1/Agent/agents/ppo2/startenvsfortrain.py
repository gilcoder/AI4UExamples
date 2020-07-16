import os
import time

input_port = 7070
output_port = 8080
host = "127.0.0.1"
sep = os.sep
for i in range(8):
    path = "%shome%skaos%sHeadWorldRC%sHeadWorldRC.x86_64 --ai4u_inputport %d --ai4u_outputport %d --ai4u_remoteip %s --ai4u_timescale 10 --ai4u_targetframerate 1000 --ai4u_vsynccount 0 -nographics -batchmode &"%(sep, sep, sep, sep, input_port + i, output_port + i, host)
    os.system(path)
    time.sleep(1)
