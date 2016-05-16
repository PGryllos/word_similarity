import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.externals import joblib

letters_dict = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8,
        'i': 9,
        'j': 10,
        'k': 11,
        'l': 12,
        'm': 13,
        'n': 14,
        'o': 15,
        'p': 16,
        'q': 17,
        'r': 18,
        's': 19,
        't': 20,
        'u': 21,
        'v': 22,
        'w': 23,
        'x': 24,
        'y': 25,
        'z': 26,
        }


git_commands = ['add', 'commit', 'push', 'pull', 'branch', 'checkout', 'reset']

test = {'add': ['ass', 'aff', 'aad', 'adf', 'addx', 'aas', 'dd', 'aaa'],
        'commit': ['cocmit', 'cocmt', 'ccommt', 'coomt', 'cpomt', 'commi'],
        'push': ['puid', 'pudh', 'pusha', 'pusht', 'psush', 'psuhj', 'oush'],
        'pull': ['pusl', 'pul', 'poll', 'pill', 'oull', 'pyll'],
        'branch': ['branc', 'brnaj', 'brnch', 'beanch', 'nranch'],
        'checkout': ['chekkout', 'cjeckout', 'checkou', 'xheckout'],
        'reset': ['rser', 'rese', 'reswt', 'seset', 'resety']}

def from_word_to_values(word, letters_dict, max_word_len):
    code = []
    for letter in word:
        code.append(letters_dict[letter])
    for i in range(len(code), max_word_len):
        code.append(0)
    return code

# +1 to capture the case when the biggest word is spelled with 1 extra letter
max_word_len = len(max(git_commands, key=len)) + 1
features = [str(i+1) + '_letter' for i in range(max_word_len)]
features.append('word')

train_dataset = pd.DataFrame(columns=features)
test_dataset = pd.DataFrame(columns=features)

# creating training dataset
train_dataset['word'] = git_commands
train_dataset[features[:-1]] = map(
        lambda word: from_word_to_values(word, letters_dict, max_word_len),
        git_commands)

print train_dataset

x_train = train_dataset[features[:-1]]
y_train = train_dataset['word']

# train several classifiers
svm_rbf = SVC(verbose=True).fit(x_train, y_train)
svm_linear = LinearSVC(verbose=True).fit(x_train, y_train)
random_forest = RF().fit(x_train, y_train)
print random_forest

# creating test dataset
for command in test:
    for wrong_word in test[command]:
        row = from_word_to_values(wrong_word, letters_dict, max_word_len)
        row.append(command)
        test_dataset.loc[len(test_dataset)] = row

x_test = test_dataset[features[:-1]]
y_test = test_dataset['word']

# test prediction accuracy
print 'SVM with rbf kernel score:', svm_rbf.score(x_test, y_test)
print 'SVM with linear kernel score:', svm_linear.score(x_test, y_test)
print 'Random Forest score:', random_forest.score(x_test, y_test)

# with that training the random forest classifier seems to outperform the other
# two models, which seems normal considering the lack of a well thought method
# for taking the words to a numeric vector space.

# you can use joblib.dumps to serialize the model of your preference to file
