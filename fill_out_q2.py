# beamsearch maxent.sh test_data boundary_file model_file sys_output beam_size topN topK

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
	top_k = str(top_k[index])
	sys_file = top_n + "_" +top_k+ ".out"
	
	command = "time `/.beamsearch_maxent.sh "+ test_data +" "+ boundary_file +" "+ model_file +" "+ sys_file +" "+ beam_size +" "+ top_n +" "+ top_k
	output = os.popen(command).read().split()
	accuracy = output[-7]
	time = output[-5]
	accuracies.append(accuracy)
	running_times.append(time)
	
output_file.write("beam size","top n","top k", "test accuracy", "running time\n")
for index in range(0,len(beam_sizes)):
	output_file.write(beam_sizes[index],top_ns[index],top_ks[index],test_accuracies[index],running_times[index],"\n")
	
output_file.close()