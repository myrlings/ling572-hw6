import sys
import operator
import math

########## functions

# read in the model from the file
def get_model(model_filename):
    model_file = open(model_filename, 'r')
    all_features = set()
    all_tags = set()
    cur_class = ""
    model = {}
    for line in model_file:
        line_array = line.split()

        # check that we're not on a new class
        if line_array[0] == "FEATURES":
            cur_class = line_array[-1]
            all_tags.add(cur_class)
            model[cur_class] = {}
            continue

        if len(line_array) < 2:
            continue
        model[cur_class][line_array[0]] = line_array[1]
        all_features.add(line_array[0])
    return [model, all_features, all_tags]

# make a list out of the boundary file
def get_boundaries(boundary_filename):
    boundary_file = open(boundary_filename, 'r')
    boundaries = []
    for line in boundary_file:
        boundaries.append(int(line))

    return boundaries

# get the vectors and tags from the test file
def get_vectors(test_data_filename):
    data_file = open(test_data_filename, 'r')

    vectors = {}
    tags = set()
    all_features = set()
    vector_order = []

    for line in data_file:
        line_array = line.split()
        instance_name = line_array[0]
        vector_order.append(instance_name)
        tag = line_array[1]

        features = line_array[2::2]
        values = line_array[3::2]

        vectors[instance_name]={}
        vectors[instance_name]["_tag_"] = tag
        tags.add(tag)

        for (f,v) in zip(features, values):
            vectors[instance_name][f] = v
            all_features.add(f)

    return [vectors, tags, vector_order, all_features]

# with the given vector, model, and set of tags
# return P(tag|vector)
def get_py_x(vector, model, tags):
    Z = 0.0
    result = {}
    for label in tags:
        summation = 0.0
        lambda_0 = 0.0
        if "_const_" in model:
            lambda_0 = float(model["_const_"])
        else:
            lambda_0 = float(model[label]["<default>"])
        for feature in vector:
            if feature == "_tag_":
                continue
            if "_const_" in model:
                    summation+=float(model["_const_"])
            elif feature in model[label]:
    		    summation+=float(model[label][feature])
        result[label] = math.e**summation
        Z += result[label]

    for label in tags:
        result[label] = result[label]/Z

    return result

# perform tag determination for each vector
# using beam search
def maxent_search(tags, vectors, model, word_order, features, boundaries\
, topN, topK, beam_size):
    answers = {}

    # sequences should store all the nodes for the current column
    sequences = {}
    
    index = 0
    if len(boundaries) > 0:
        cur_boundary = boundaries.pop(0)
    for vector in word_order:
        index += 1

        # at BOS, reset things
        if index == cur_boundary + 1 or vector == word_order[0]:
            if (vector != word_order[0]):
                word_ind = word_order.index(vector) - 1
                add_answers(answers, sequences, word_order[word_ind], cur_boundary,\
                word_order)
                cur_boundary = boundaries.pop(0)
            index = 1
            sequences = {}
            sequences[index] = {}
            
            # construct vector with prev two tags being BOS
            temp_vector = vectors[vector]
            temp_vector["prevT=BOS"] = 1
            temp_vector["prev2T=BOS+BOS"] = 1
            result = get_py_x(temp_vector, model, tags)
            #print vector, temp_vector, result

            # get a list of keys from result corresponding to 
            # the topN best-prob tags
            topNtags = sorted(result.iteritems(), key=operator.itemgetter(1),\
            reverse=True)
            topNtags = topNtags[0:topN]

            for tag in topNtags:
                sequences[index][tag[0]] = result[tag[0]]
            #print topNtags, sequences
        else:
            sequences[index] = {}
            # store sequences for the previous column, then reset sequences
            # to look only at the current column
            for seq in sequences[index-1]:
                # form vector 
                tag_list = seq.split()
                if len(tag_list) == 1:
                    ti_1 = seq
                    ti_2 = "BOS"
                else:
                    ti_1 = tag_list[len(tag_list)-1]
                    ti_2 = tag_list[len(tag_list)-2]
                prevT = "prevT=" + ti_1
                prev2T = "prevTwoTags=" + ti_1 + "+" + ti_2
                temp_vector = vectors[vector]
                if prevT in features: temp_vector[prevT] = 1
                if prev2T in features: temp_vector[prev2T] = 1
                
                # find probabilities with the vector we made
                result = get_py_x(temp_vector, model, tags)
                #print vector, temp_vector, result
                
                topNtags = sorted(result.iteritems(), key=operator.itemgetter(1),\
                reverse=True)
                topNtags = topNtags[0:topN]

                # store new nodes for current column
                for tag in topNtags:
                    temp_seq = seq + " " + tag[0]
                    temp_prob = result[tag[0]]
                    sequences[index][temp_seq] = sequences[index-1][seq] *\
                    temp_prob
            # perform pruning
            sequences[index] = prune(sequences[index], topK, beam_size)

    #print "adding:", vector, cur_boundary, word_order
    add_answers(answers, sequences, vector, cur_boundary,\
    word_order)
    #print sequences 
    return answers

# find the winning sequence, and decipher it to store in answer
def add_answers(answers, sequences, vector, boundary, word_order):
    #print sequences
    # find the best sequence at the final column
    highest_index = max(sequences.keys())
    best_seq = max(sequences[highest_index].iteritems(), key=operator.itemgetter(1))[0]

    # split up the sequence, add the answer for each column
    seq_list = best_seq.split()
    word_order_n = word_order.index(vector) - boundary + 1
    # be sure this is the beginning word of the current sentence
    #print word_order[word_order_n]
    for i in range(len(seq_list)):
        cur_tag = seq_list[i]
        cur_seq = ""
        # kind of a terrible way to get what sequence we want the probability
        # of, but it works
        for j in range(i+1):
            cur_seq += seq_list[j] + " "

        cur_prob = sequences[i+1][cur_seq[0:-1]]
        cur_word = word_order[word_order_n+i]
        answers[cur_word] = [cur_tag, cur_prob]

# perform pruning wrt topK and beam_size
def prune(sequences, topK, beam_size):
#    print "before:", sequences
    max_prob = max(sequences.values())
    topK_nodes = [] # store topK tuples with tag, prob
    for seq in sequences:
        cmp1 = math.log(sequences[seq]) + beam_size
        # check beam size
        if cmp1 < math.log(max_prob):
            continue
        # add until we get to topK
        elif len(topK_nodes) < topK:
            topK_nodes.append((seq, sequences[seq]))
        # we should add a new sequence to topK_nodes
        elif sequences[seq] > topK_nodes[topK-1][1]:
            topK_nodes = insert_seq(topK_nodes, seq, sequences[seq])
    new_sequences = {}
    for node in topK_nodes:
        new_sequences[node[0]] = node[1]

#    print "after:", new_sequences
    return new_sequences

# put the probability for the sequence in its proper place among topK_nodes
def insert_seq(topK_nodes, seq, prob):
    new_list = []
    for slot in range(len(topK_nodes)):
        if prob > topK_nodes[slot][1]:
            tup = (seq, prob)
            if slot == 0:
                new_list.append(tup)
                new_list.extend(topK_nodes[0:len(topK_nodes)])
            else:
                new_list = topK_nodes[0:slot]
                new_list.append(tup)
                new_list.extend(topK_nodes[slot:len(topK_nodes)])
            break
    return new_list[0:len(topK_nodes)]
            

# output results to sys file
def print_sys(sys_filename, answers, vectors, word_order):
    sys_file = open(sys_filename, 'w')
    sys_file.write("%%%%%%% test data:\n")

    for word in word_order:
        sys_file.write(word + " ")
        sys_file.write(vectors[word]["_tag_"] + " ")
        sys_file.write(answers[word][0] + " " + str(answers[word][1]))
        sys_file.write("\n")

# print confusion matrix
def print_acc(tags, answers, vectors, features):
    print "class_num=", len(tags), ", feat_num=", len(features) 

    print "\nConfusion matrix for the testing data:"
    print "row is the truth, column is the system output\n"
    counts = {}
    num_right = 0
    for actualtag in tags:
        sys.stdout.write("\t" + actualtag)
        counts[actualtag] = {}
        for expectedtag in tags:
            counts[actualtag][expectedtag] = 0
    for word in vectors:
        actual_tag = vectors[word]["_tag_"]
        expected_tag = answers[word][0]
        counts[actual_tag][expected_tag] += 1
        if actual_tag == expected_tag:
            num_right += 1

    sys.stdout.write("\n")
    for actualtag in tags:
        sys.stdout.write(actualtag)
        for expectedtag in tags:
            sys.stdout.write("\t" + str(counts[actualtag][expectedtag]))
        sys.stdout.write("\n")
    accuracy = float(num_right) / len(vectors)
    sys.stdout.write("Test accuracy: ")
    sys.stdout.write(str(accuracy))


########## main
if len(sys.argv) < 8:
    print "Format is: beamsearch_maxent.sh test_data boundary_file model_file sys_output beam_size topN topK"
    sys.exit()

test_filename = sys.argv[1]
boundary_filename = sys.argv[2]
model_filename = sys.argv[3]
sys_filename = sys.argv[4]
beam_size = int(sys.argv[5])
topN = int(sys.argv[6])
topK = int(sys.argv[7])

# store model and set of features
model_features = get_model(model_filename)
model = model_features[0]
features = model_features[1]
boundaries = get_boundaries(boundary_filename)
tags = model_features[2]

# store vectors, set of all tags
vectors_tags = get_vectors(test_filename)
vectors = vectors_tags[0]
word_order = vectors_tags[2]


# beam search
answers = maxent_search(tags, vectors, model, word_order, features,\
boundaries, topN, topK, beam_size)
print_sys(sys_filename, answers, vectors, word_order)
print_acc(tags, answers, vectors, features)
