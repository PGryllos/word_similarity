import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.externals import joblib

# 1st and most naive case of mapping. No intuition behind the mapping.
# SVM with rbf kernel score: 0.609756097561
# SVM with linear kernel score: 0.439024390244
# Random Forest score: 0.853658536585
letters_dict_1 = {
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


# 2nd case of mapping. Trying to give similarly close values to keys that are
# in similar distances from a starting point (button a).
# SVM with rbf kernel score: 0.780487804878
# SVM with linear kernel score: 0.634146341463
# Random Forest score: 0.853658536585
letters_dict_2 = {
        'a': 2,
        'b': 15,
        'c': 9,
        'd': 8,
        'e': 7,
        'f': 11,
        'g': 14,
        'h': 17,
        'i': 23,
        'j': 20,
        'k': 22,
        'l': 25,
        'm': 21,
        'n': 16,
        'o': 24,
        'p': 26,
        'q': 1,
        'r': 12,
        's': 5,
        't': 13,
        'u': 19,
        'v': 10,
        'w': 6,
        'x': 4,
        'y': 18,
        'z': 3,
        }


# 3rd case of mapping. Second try to capture button closeness numerically.
# SVM with rbf kernel score: 0.780487804878
# SVM with linear kernel score: 0.658536585366
# Random Forest score: 0.878048780488
letters_dict_3 = {
        'a': 2,
        'b': 9,
        'c': 6,
        'd': 5,
        'e': 4,
        'f': 5.5,
        'g': 8,
        'h': 8.5,
        'i': 10.5,
        'j': 11,
        'k': 11.5,
        'l': 14,
        'm': 12,
        'n': 9.5,
        'o': 13,
        'p': 13.5,
        'q': 1,
        'r': 4.5,
        's': 2.5,
        't': 7,
        'u': 10,
        'v': 6.5,
        'w': 1.5,
        'x': 3.5,
        'y': 7.5,
        'z': 3,
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
        lambda word: from_word_to_values(word, letters_dict_3, max_word_len),
        git_commands)

print train_dataset

x_train = train_dataset[features[:-1]]
y_train = train_dataset['word']

# train several classifiers
svm_rbf = SVC(verbose=True).fit(x_train, y_train)
svm_linear = LinearSVC(verbose=True).fit(x_train, y_train)
random_forest = RF(n_estimators=1000).fit(x_train, y_train)
print random_forest

# creating test dataset
for command in test:
    for wrong_word in test[command]:
        row = from_word_to_values(wrong_word, letters_dict_3, max_word_len)
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

# you can use joblib.dump to serialize the model of your preference to file
