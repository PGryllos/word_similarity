### How hard can word similarity be (?)

This is just an afternoon project. I tried to prototype something similar to `git's` word suggestion mechanism.
For example, when you accidentally type `git ads -p` instead of `git add -p` and git prompts you this message
```
git: 'ads' is not a git command. See 'git --help'.

Did you mean this?
	add

```
have you ever wondered how hard can this mechanism be? Simple or not, it seems quite fancy so I though I should give it a try. 

#### note
To be clear, this not a suggested approach that aims to outperform any well-established technique or anything like that. I didn't use any IR literature or any well known Distance for such problems (like the Levenshtein distance that git uses for that). My aim was to mess around with simple classification techniques intuitively (or non-intuitively) and test how well such a predictor can perform with minimum effort.

#### problem
So, let's say you have an application that relies on a command-line interaction with the user and you want to provide
the user with suggestions about the command he / she was trying to type when he accidentally typed something else (e.g. 'ads' instead of 'add' or 'ccomit' instead of 'commit').

The problem is quite straightforward. I created a script where someone can define a set of commands (e.g. `my_apps_commands = ['conduct', 'downsample', 'dogge']`) and with minimum training can create a model that used in his / her app will give a fairly good prediction about what the user was trying to type.

####  the method
The outline of the method I followed is as simple as it gets. Represent each word in a feature space and train several classifiers. Meassure the performance of each classifier and see there is a chance for production of meaningfull results.  

The numerical representation I choose for transfering the words to a feature space is also the simplest thing I could come up with. Create a feature space where there is a feature for each letter of the biggest word you want your app to be able to classify. Break the word to letters and assign a real value to each letter through a standar mapping. Let's say `a==1`, `b==2`, `c==3` etc. That means that if we want to classify two commands `add, rm` the representation for these words
would be

letter_1 | letter_2 | letter_3 | word
-------- | -------- | -------- | ----
1        | 4        | 4        | add
18       | 13       | 0        | rm

I used the convention that value 0 means that this word doesn't have that many letters. As `rm` doesn't have a 3rd letter.

#### results
I think I was correct. Prototyping something like that mechanism is not that hard. Taking into consideration the scores I the classifiers achieved. Here is an output: 
```
SVM with rbf kernel score: 0.609756097561
SVM with linear kernel score: 0.439024390244
Random Forest score: 0.80487804878
```
Have in mind that this method is not able to detect any missalignment on the spelled words. Maybe a dtw approach where each word is considered a time variant signal could improve on those results. Something that must be also noted, and obviously lessens the power of that method, is that although a numeric vector is constructed for each for each word, those numeric values have only categorical meaning. Meaning that for example the value `1` states that there is a letter a in that position. But, the closeness of the values `1` and `2` doesn't not apply to the closeness of `a` and `b`. Maybe an improvement to the existing technique would be to assign values numerically close to letters that tend to be frequently misspelled for one another.

#### how to use
In the `word_similarity.py` you can see the words I used for the training and the misspelled words I used for measuring the performance of the classifiers. You can define your own set of words you want to use as commands and serialize to a file for use in your application.
