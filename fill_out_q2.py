# beamsearch maxent.sh test_data boundary_file model_file sys_output beam_size topN topK
import os

beam_sizes = [0,1,2,3]
top_ns = [1,3,5,10]
top_ks = [1,5,10,100]
test_accuracies = []
running_times = []

output_file = open("q2.txt","w")


test_data = "sec19_21.txt"
model_file = "m1.txt"
boundary_file = "sec19_21.boundary"

for index in range(0,len(beam_sizes)):
	beam_size = str(beam_sizes[index])
	top_n = str(top_ns[index])
	top_k = str(top_ks[index])
	sys_file = top_n + "_" +top_k+ ".out"
	
	command = "{ time ./beamsearch_maxent.sh "+ test_data +" "+ boundary_file +" "+ model_file +" "+ sys_file +" "+ beam_size +" "+ top_n +" "+ top_k + ";} 2> temp"
	output = os.popen(command).read().split()
	accuracy = output[-1]
	time = open("temp","r").readlines()[1].split("\t")[1].strip("\n")
	test_accuracies.append(accuracy)
	running_times.append(time)
	os.popen("rm temp")
	

for index in range(0,len(beam_sizes)):
	output_line = "beam_size="+str(beam_sizes[index])+"\t"+"top_n="+str(top_ns[index])+"\t"+"top_k="+str(top_ks[index])+"\t"+"test_accuracy="+str(test_accuracies[index])+"\t"+"running_time="+str(running_times[index])+"\n"
	output_file.write(output_line)
	
output_file.close()