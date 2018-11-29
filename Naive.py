import os
import copy,math

#folder_path is the path of parent directory containing 20 categories
folder_path = '20_newsgroups/'
list_of_folders = os.listdir(folder_path)
train_data = 500
master_files = {} #dictionary of trained files under each category

master_word_freq = {} #a master dictionary of total words.
dic_of_dict = {} #dictionary of words under each category 
category = 20
alpha = 0.0001 #laplace smoothing

def data_handler(text):
    spl_characters = ['~','`','!','@','^','$','%','&','*','(',')','+','=','{','}','[',']',';',':','|','\\','"',"'",'\n','<','>',',','.','?','/','-','*']
    for i in spl_characters:
        text = text.replace(i , ' ')
    text = text.lower()
    #stop words are eliminated for more accurate prediction
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    for j  in stop_words:
        text = text.replace(j, '')
    return text

print('Training the first 500 files')
print('Processing the text.........')
for folder in list_of_folders:
    mapper = {}#dic
    count = 0
    list_of_files = os.listdir(folder_path + folder)
    for file in list_of_files:
        if count < train_data:
            file_path = folder_path + folder + '/' + file
            curr_file = open(file_path, 'r')
            file_data = data_handler(curr_file.read())
            for word in file_data.split(' '):
                word = word.strip()
                if word != ' ' and word != '':
                    if mapper.get(word,0) == 0:
                        mapper[word] = 1
                    else:
                        mapper[word] = mapper.get(word,0) + 1
                    if master_word_freq.get(word,0) == 0:
                        master_word_freq[word] = 1
                    else:
                        master_word_freq[word] = master_word_freq.get(word,0) + 1
            list_of_files.remove(file)
        count= count+1
    master_files[folder] = list_of_files
    dic_of_dict[folder] = mapper
print('Total number of words found in the dataset : ' + str(len(master_word_freq)))
print('Testing the remaining 500 files')
test_folders = copy.deepcopy(list_of_folders)
probabilities = []
res_dir = {}
j=0
print('Calculating probabilities.........')
for folder in test_folders:
    res_dir[folder] = 0
for folder in test_folders:
    t_list_of_files = os.listdir(folder_path + folder)
    flag =0
    prob1 = 0.0
    for file in t_list_of_files:
        if file not in master_files[folder]:
            flag+=1
            file_path = folder_path + folder + '/' + file
            curr_file = open(file_path, 'r')
            file_data = data_handler(curr_file.read())
            sum1 = sum(dic_of_dict[folder].values())
            for word in file_data.split(' '):
                if word != ' ' and word != '' and word not in dic_of_dict[folder].keys():
                    word = word.strip()
                    prob2 = float(dic_of_dict[folder].get(word, 0.0)) + alpha
                    prob1 = prob1 +(math.log(float(prob2)/float(sum1)) )
                    #probabilities.append(prob1)
            t_list_of_files.remove(file)
    probabilities.append(prob1)
    res_dir[test_folders[probabilities.index(max(probabilities))]] += 1
frequency = list(res_dir.values())
maxi = max(frequency)
accuracy = maxi/category*100
print("Accuracy : " + str(accuracy) + "%")    