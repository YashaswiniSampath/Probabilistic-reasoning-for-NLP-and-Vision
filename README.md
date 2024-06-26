# Probabilistic-reasoning-for-NLP-and-Vision

## **Part 1: NLP Pos Tagging**

One of the fundamental challenges in NLP is part-of-speech tagging, where the aim is to label each word in a sentence with its corresponding part of speech (e.g., noun, verb, adjective). This is a crucial step towards extracting semantics from natural language text. By utilizing Bayes Nets, Simplified Algorithms, Variable Elimination Algorithms, and the Viterbi Algorithm, we can predict the tags for each word in an input sentence.

In this project, we have tried achieving results using various algorithms like Simplified and Viterbi Algorithm for predicting the tags of input sentences.

~~~
Example for train dataset
Poet NOUN twisted VERB again ADV and CONJ Nick's NOUN knuckles NOUN scraped VERB on ADP the DET air NOUN tank NOUN , . ripping VERB off PRT the DET skin NOUN . .   
Desperately ADV , . Nick NOUN flashed VERB one NUM hand NOUN up PRT , . catching VERB Poet's NOUN neck NOUN in ADP the DET bend NOUN of ADP his DET elbow NOUN . .   
The DET air NOUN hose NOUN was VERB free ADJ ! . ! .  
~~~

### (1)Description of how you formulated each problem; 

There are two parts to it:
1. Simplified 
2. HMM

a. Simplified:
The simplified model is defined such that the state to be predicted(hidden state) is not dependent of its previous states but only depends on the emission probability (P(observed state| hidden state) which in this case is P(word | POS)) and the P(hidden state) which is P(POS) here.
We simply calculate the probabilities and find the max of the probabilities of all the hidden states and POS with the max probability is the result for that particular observed state.

b. HMM:
The HMM is based on the (maximum a posteriori (MAP)), we populate the Viterbi dictionary by a tuple(prob,[output List]).

**Calculating the probability:**
Part1: Initial Prob of the 1st row is calculated using emission probability for the current observation * Initialprob[observed]
    Initial probability is the occurrence of a POS at the start of a sentence. and then we find the probability of it (normalize it)
Part2: For length 1 to len(sentence)
    prob = max(transition prob from previous layer) * emission probability for the current observation * viterbiDictionary value of previous state(viterbi[prev_tag])
    choose the state with maximum prob, add the POS into the output list
Once all observations are processed, choose the state with the highest final Viterbi path probability as the most likely end state.
Return the output List

### (2) A brief description of how your program works; 
The computation can be divided into three sections:
    1.0 Initial Probability
    1.1 Emission Probability
    1.2 Transition Probability
    1.3 Viterbi algorithm

**1.0 Initial probability:**
We calculate the probability that the first word of the sentence is a particular POS in the self.dict_pos.

**1.1 Emission probability:**
Emission probability is calculated by P(observed word|POS)
For this, we go through each word and store all the words of a particular POS in the emmisionProb dictionary and maintain a count.
We then normalize this to get the proper probability values.

**1.2 Transition Probability:**
Transition probability is calculated by P(Ti|Ti-1):
Transition Prob dictionary is a number of POS * number of POS (12 x 12)
1st pass: calculate the freq of occurrence of a POS given prev POS in the train file
2nd pass: normalize it to find the probability

**1.3 Viterbi Algorithm**
prob = max(transition prob from previous layer)* emission probability for the current observation * viterbiDictionary value of previous state(viterbi[prev_tag])

### (3) Discussion of any problems you faced, any assumptions, simplifications, and/or design decisions you made.

1. The accuracy was coming out to be quite low in certain scenarios when the word was not present in the train set. For such situations, we have hardcoded a very small value = 10^-10. This has helped increase the accuracy where the probability does not directly drop to 0 and simply returns the last POS in the POS_list.
2. Accuracy further improved when the assumption was made that the tag to be predicted to be set as Noun. This was one of the design decisions as the occurrence of nouns in a sentence is quite decent. And it handles the case when a Proper Noun which has a probability of not being in the trainset to be classified accurately.
3. We also take care of sentences which has only 1 word.

### Result

So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       92.91%               43.15%
            2. HMM:       95.02%               54.00%



## **Part 2: Optical Character Recognition (OCR) **
The purpose of this project is to develop an Optical Character Recognition (OCR) system capable of accurately recognizing and extracting text from noisy images (include images with noise up to 80%).

### (1)Description of how you formulated each problem; 
For each test character, the model calculates the probability it belongs to each character in the training set (TRAINLETTERS) and selects the one with the highest probability. 
Training Image:
<img width="1014" alt="Screenshot 2024-06-26 at 8 57 45 PM" src="https://github.com/YashaswiniSampath/Probabilistic-reasoning-for-NLP-and-Vision/assets/44898518/8a0936cb-aab5-45cf-b816-c52342c4c10b">

Few Test Images:
<img width="806" alt="Screenshot 2024-06-26 at 9 01 10 PM" src="https://github.com/YashaswiniSampath/Probabilistic-reasoning-for-NLP-and-Vision/assets/44898518/c9b985e9-a634-4108-acd5-4f25b393dc95">
<img width="998" alt="Screenshot 2024-06-26 at 9 06 03 PM" src="https://github.com/YashaswiniSampath/Probabilistic-reasoning-for-NLP-and-Vision/assets/44898518/d7e18f13-3d14-40ff-92b6-a2cf76e4eece">

There are two parts to it:
1. Simplified: Utilizes a straightforward method based on emission probabilities to predict the characters in the input images.
2. HMM : Employs a more sophisticated approach using the Viterbi algorithm to consider both emission and transition probabilities, thereby improving the accuracy of character sequence predictions.

a.Simplified:
The simplified can be predicted by just considering the emission probability matrix.
and returning the highest probability in that row. i.e given a test character the probability it belongs to one in TRAINLETTERS

b. HMM:
The HMM is based on the (maximum a posteriori (MAP)), we populate the Viterbi dictionary by a tuple(prob,[output List]).
**Calculating the probability:**
Part1: Initial Prob of the 1st row is calculated using emission probability for the current observation * Initialprob[observed]
    Initial probability is the occurrence of word[i] at the start of a sentence. and then we find the probability of it(normalize it)
Part2: For length 1 to len(test letters)
    prob = max(transition prob from previous layer)* emission probability for the current observation * viterbiDictionary value of previous state(viterbi[prev_char])
    choose state with maximum prob, add the character into the output list
Once all observations are processed, choose the state with the highest final Viterbi path probability as the most likely end state.
Return the output List

### (2) A brief description of how your program works; 
The computation can be divided into three sections:
    1.1 Emission Probability
    1.2 Transition Probability
    1.3 Viterbi algorithm

**1.1 Emission probability:**
Emission probability is calculated by P(observerdTestImage|GivenTrainImage)
Since some images are noisy we try to differentiate them by assigning different weights based on the density level of images. Initially, we run both train and test images which help us determine the density % based on the no of pixels denoted by "*".
Emissionprob dictionary is a len(testImagelen)*TRAIN_LETTERS.
We have 4 conditions when evaluating two images:
    a.If both pixels match: eg * and *
    b.If both blank spaces match: " and "
    c.If the pixel is present in the train but not in the test
    d.if the pixel is not present in the train but in the test, this indicates noise

Each of the above counts is calculated after evaluation of different weights we have assigned, after different permutations and combinations of calculation probability like using: math.log to minimize the computation
math.pow
assigning weights

**1.2 Transition Probability:**
Transition probability is calculated by P(Ti|Ti-1):
Transition Prob dictionary is a TRAIN_LETTERS * TRAIN LETTERS
1st pass: calculate the freq of occurrence of a char given prev char in the train file
2nd pass: normalize it to find the probability

**1.3 Viterbi Algorithm**
prob = max(transition prob from previous layer)* emission probability for the current observation * viterbiDictionary value of previous state(viterbi[prev_char])

### (3) Discussion of any problems you faced, any assumptions, simplifications, and/or design decisions you made.

1. Emission probability: Tried calculating the probability by just calculating the match count i.e. if train[h][w]==test[h][w] it performed very badly as it matched many spaces. Later replaced that by considering different weights for different comparisons (such as pixels, and blanks).
2. Designing the dictionary for Viterbi algo computation, we designed it to store a tuple of prob and output list
3. The Emission probability calculation after different permutations and combination of calculation probability like using : 
        math.log to minimize the compute
        math.pow
        assigning weights
