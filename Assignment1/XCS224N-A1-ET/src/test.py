import numpy as np
from sklearn.decomposition import TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN  = '<END>'

# sample text
# test_corpus = ["{} My name is Priya {}".format(START_TOKEN,END_TOKEN).split(" "),"{} Priya is good {}".format(START_TOKEN,END_TOKEN).split(" "),
#                "{} My name is Viha {}".format(START_TOKEN,END_TOKEN).split(" "),"{} My name is Vijay {}".format(START_TOKEN,END_TOKEN).split(" ")]
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN,END_TOKEN).split(" "),"{} All's well that ends well {}".format(START_TOKEN,END_TOKEN).split(" ")]
# print(test_corpus)

#find distinct words
#list comprehension
# val = [i*2 for i in range(10)]
# print(val)
def distinct_words(corpus):
    distinct_words = sorted(set([word for line in corpus for word in line]))
    len_distinct = len(distinct_words)
    # print(distinct_words)
    # print(len_distinct)
    return distinct_words,len_distinct

##### Distinct END ######
# find co-occurence matrix based on certain window size = 4
# line = "My name is priya"
# win_size = 2
# line = "A,B,C,D,E,F"
# win_size = 4
# line_list = line.split(",")
# print(line_list)
# for i,word in enumerate(line_list):
#     # print(i)
#     start_index = max(i - win_size,0)
#     end_index = min(i + 1 + win_size,len(line_list))
#     print(line_list[start_index:i],line_list[i],line_list[i+1:end_index])

# if given with more than one statement
# line = """My name is priya
# My name is Viha
# My name is Vijay
# """
win_size = 1
# # line_list = line.replace("\n"," ").split(" ")
# line_list = line.split("\n")
# print(line_list)
words_list, distinct_count = distinct_words(test_corpus)
#convert to dict
# word2ind = [{word:index} for index,word in enumerate(words_list)]
word2ind = {word:index for index,word in enumerate(words_list)}
R = np.zeros((distinct_count,distinct_count))
# print(R)
# print(word2ind)
test_corpus = test_corpus[0:2]
for line in test_corpus:
    line_len = len(line)
    #instead of words - link its respective index
    # print(line)
    line_ids = [word2ind[word] for word in line ]
    # print(line_ids)
    for index in range(line_len):
        startindex = max(index - win_size,0)
        endindex = min(index + win_size + 1,line_len)
        # print(line[startindex:index],line[index],line[index+1:endindex])
        # print(line_ids[startindex:index],line_ids[index],line_ids[index+1:endindex])
        center_id = line_ids[index]
        combine_left_right = line_ids[startindex:index]+line_ids[index+1:endindex]
        for word_id in combine_left_right:
            R[center_id,word_id] +=1
print(R,word2ind)

#write SVD ()
iters = 4
M_reduced = None
print("Running Truncated SVD over %i words...." %(R.shape[0]))

#run truncate SVD
svd = TruncatedSVD(n_components=2, n_iter=iters)
svd.fit(R)
M_reduced =  svd.transform(R)
print("Done.")
print(M_reduced)

    