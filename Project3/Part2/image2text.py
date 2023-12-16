#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: Avishmita Mandal - avmandal
#          Yashaswini Sampath Kumar - ysampath

import copy
import math
import sys

from PIL import Image, ImageDraw, ImageFont

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(
        0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH
    ):
        result += [
            [
                "".join(
                    [
                        "*" if px[x, y] < 1 else " "
                        for x in range(x_beg, x_beg + CHARACTER_WIDTH)
                    ]
                )
                for y in range(0, CHARACTER_HEIGHT)
            ],
        ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    )
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


# Calculate the emission prob: P(observedState|HiddenState)
def calculate_emission(train_letters, test_letters):
    TRAIN_LETTERS = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    )
    # Dictionary to store the emission probability
    emissionProbability = {
        tag: {othertag: 0 for othertag in train_letters}
        for tag in range(len(test_letters))
    }
    # To determine the density of noise in test and train images
    starCountInTest = 0
    starCountInTrain = 0
    for letter in test_letters:
        for i in letter:
            if i == "*":
                starCountInTest += 1
    for letter in train_letters:
        for i in train_letters[letter]:
            if i == "*":
                starCountInTrain += 1
    # Traverse and update emission prob
    for i in range(len(test_letters)):
        for j in TRAIN_LETTERS:
            starCount, blankCount, blacknoisecount, noiseCount = 0, 0, 0, 0
            for h in range(CHARACTER_HEIGHT):
                for w in range(CHARACTER_WIDTH):
                    # If we encounter a star to be expected in both train and test
                    if (
                        test_letters[i][h][w] == train_letters[j][h][w]
                        and test_letters[i][h][w] == "*"
                    ):
                        starCount += 1
                    # If we encounter a blank space to be expected in both train and test
                    elif (
                        test_letters[i][h][w] == train_letters[j][h][w]
                        and test_letters[i][h][w] == " "
                    ):
                        blankCount += 1
                    # If the expected pixel is not present in test
                    elif train_letters[j][h][w] == "*":
                        blacknoisecount += 1
                    # If additional noise is present in train
                    else:
                        noiseCount += 1

                # Assign weights according to the noise density
                if starCountInTest / len(test_letters) > starCountInTrain / len(
                    train_letters
                ):
                    emissionProbability[i][j] = (
                        math.pow(0.8, starCount)
                        * math.pow(0.7, blankCount)
                        * math.pow(0.3, blacknoisecount)
                        * math.pow(0.2, noiseCount)
                    )
                else:
                    emissionProbability[i][j] = (
                        math.pow(0.99, starCount)
                        * math.pow(0.7, blankCount)
                        * math.pow(0.3, blacknoisecount)
                        * math.pow(0.01, noiseCount)
                    )
    return emissionProbability


# Simplified Algo
def simplified(train_letters, test_letters):
    TRAIN_LETTERS = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    )
    emissionProbabilty = calculate_emission(train_letters, test_letters)
    # List for storing the results based on emission prob
    list = []
    for i in range(len(test_letters)):
        max_prob = -sys.maxsize
        for j in TRAIN_LETTERS:
            if j in emissionProbabilty[i]:
                if emissionProbabilty[i][j] >= max_prob:
                    max_prob = emissionProbabilty[i][j]
                    char = j
        list.append(char)
    return "".join(list)


# HMM Algo
def HMM(train_letters, test_letters):
    smallEmissionProb = pow(10, -10)
    TRAIN_LETTERS = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    )
    emissionProbability = calculate_emission(train_letters, test_letters)
    # Dictionary for storing transitionProb P(ti|ti-1)
    transitionProb = {
        tag: {othertag: 0 for othertag in train_letters} for tag in train_letters
    }
    file = open(TrainDataset, "r")
    for line in file:
        for i in range(len(line) - 1):
            prev_state = line[i]
            current_state = line[i + 1]
            if prev_state in transitionProb and current_state in transitionProb:
                transitionProb[prev_state][current_state] += 1  # Store the count

    # Normalizing to get Transition Probability
    for tag in transitionProb:
        row_sum = sum(transitionProb[tag].values())
        if row_sum > 0:
            for other_tag in transitionProb[tag]:
                transitionProb[tag][other_tag] /= row_sum
                if transitionProb[tag][other_tag] == 0:
                    transitionProb[tag][other_tag] = smallEmissionProb
        else:
            # If the tag isnt present in traindata,we assign smallEmissionProb instead of 0
            for other_tag in transitionProb[tag]:
                transitionProb[tag][other_tag] = smallEmissionProb

    # Hashmap for storing initial prob of char
    initialProb = {}
    for x in range(len(TRAIN_LETTERS)):
        initialProb[TRAIN_LETTERS[x]] = 0

    # Calculate initial Prob count
    file = open(TrainDataset, "r")
    for line in file:
        firstchar = line[0]
        if firstchar in initialProb:
            initialProb[firstchar] += 1

    # Normalize to get initial Probability values
    total = sum(initialProb.values())
    for char in initialProb:
        if initialProb[char] > 0:
            initialProb[char] = math.log(initialProb[char] / total)

    # Create a dictionary of dictionary to store the (probabilty,[hidden char]) for the Viterbi Table
    sentence_dictionary = {
        index1: {index2: {} for index2 in train_letters}
        for index1 in range(len(test_letters))
    }

    # Handling the first character of each sentence to get initialProb
    for i in TRAIN_LETTERS:
        prob = 0
        prob += math.log(emissionProbability[0][i]) + initialProb[i]
        sentence_dictionary[0][i] = (prob, [i])

    result = []
    # Handling remaining characters from 1-len(testString)
    for i in range(1, len(test_letters)):
        prev_char = ""
        current_char = ""
        minprob = 0
        # Extra the current and previous character
        for j in TRAIN_LETTERS:
            if emissionProbability[i - 1][j] > minprob:
                minprob = emissionProbability[i - 1][j]
                prev_char = j
        minprob = 0
        for k in TRAIN_LETTERS:
            if emissionProbability[i][k] > minprob:
                minprob = emissionProbability[i][k]
                current_char = k

        # Update the sentence Dictionary by calculating emissionProb * TransitionProb * SentenceDictionary[PreviousChar]
        for trainchar in TRAIN_LETTERS:
            maxProbability = -sys.maxsize
            outputList = []
            for prevTrainChar in TRAIN_LETTERS:
                probability = 0
                # Probability if the currrent char is in emissionProb
                if current_char in emissionProbability[i]:
                    probability += (
                        math.log(
                            emissionProbability[i][trainchar]
                        )  # Using math.log() to avoid huge calculations
                        + math.log(transitionProb[prevTrainChar][trainchar])
                        + sentence_dictionary[i - 1][prevTrainChar][0]
                    )
                # else use smallEmissionProb instead of emissionProb
                else:
                    probability += (
                        math.log(smallEmissionProb)
                        + math.log(transitionProb[prevTrainChar][trainchar])
                        + sentence_dictionary[i - 1][prevTrainChar][0]
                    )
                if probability >= maxProbability:
                    maxProbability = copy.deepcopy(probability)
                    # Copy the list from prevMax probability state
                    outputList = copy.deepcopy(
                        sentence_dictionary[i - 1][prevTrainChar][1]
                    )
            outputList.append(trainchar)
            sentence_dictionary[i][trainchar] = (
                maxProbability,
                outputList,
            )  # Update sentence Dictionary
        result = copy.deepcopy(outputList)

    # Handling the last hidden layer, returning list with max probability value
    max_prob = -sys.maxsize
    result = []
    for key, (prob, list) in sentence_dictionary[len(test_letters) - 1].items():
        if prob >= max_prob:
            max_prob = prob
            result = list
    return "".join(result)


#####
# Main program
if len(sys.argv) != 4:
    raise Exception(
        "Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png"
    )

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
# Train file we are using the bc.train from Part1
TrainDataset = train_txt_fname

# Simplified
SimplifiesAns = simplified(train_letters, test_letters)

# HMM
HMMAns = HMM(train_letters, test_letters)

# The final two lines of your output should look something like this:
print("Simple: " + SimplifiesAns)
print("   HMM: " + HMMAns)
