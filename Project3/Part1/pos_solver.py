###################################
# CS B551 Fall 2023, Assignment #3
#
# Repo name : Avmandal-ysampath-a3
# Yashaswini Sampath - ysampath
# Avishmita Mandal - avmandal


import copy
import math


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def __init__(self) -> None:
        self.log_prob = []
        self.small_emmision_prob = pow(10, -10)

    def posterior(self, model, sentence, label):
        # Calculating posterior probability for simplified model - log(P(posterior)) = log(P(POS)) + log(P(POS|word))
        # For the simplified model we do not consider the transition probability
        if model == "Simple":
            simple_sum = 0
            for i in range(len(label)):
                simple_sum += math.log(self.dict_pos[label[i]])
                if sentence[i] in self.emissionProb[label[i]]:
                    simple_sum += math.log(
                        self.emissionProb[label[i]][sentence[i]]
                    )  # A very small number so that the probability does not go to 0
                else:
                    simple_sum += math.log(self.small_emmision_prob)
            return simple_sum

        # Calculating posterior probability for HMM - log(P(posterior)) = log(P(Initial_Probabilty(POS))) + log(P(POS|word)) + log(P(POS(i)|POS(i)))
        # In the Viterbi Algorithm we consider the Transition Probability
        elif model == "HMM":
            tempsum1, tempsum2 = self.small_emmision_prob, self.small_emmision_prob

            # Calculating for Initial State
            initial_sum = math.log(self.initial_prob[label[0]])
            if sentence[0] in self.emissionProb[label[0]]:
                tempsum1 = math.log(self.emissionProb[label[0]][sentence[0]])

            totalProb = initial_sum + tempsum1

            # Calculate for the remaining starting from index 1
            for i in range(1, len(sentence)):
                if sentence[i] in self.emissionProb[label[i]]:
                    tempsum1 = math.log(self.emissionProb[label[i]][sentence[i]])

                # Transition Probability
                if sentence[i] in self.transitionProb[label[i]]:
                    tempsum2 = math.log(self.transitionProb[label[i - 1]][label[i]])

            totalProb += tempsum1 + tempsum2
            return totalProb
        else:
            print("Unknown algo!")

    # Do the training!
    def train(self, data):
        # We maintain the list of all Parts of Speech
        pos = [
            "adj",
            "adv",
            "adp",
            "conj",
            "det",
            "noun",
            "num",
            "pron",
            "prt",
            "verb",
            "x",
            ".",
        ]

        # Dictionary to store the probability of a part of speech
        self.dict_pos = copy.deepcopy({dict_pos_tag: 0 for dict_pos_tag in pos})

        # Dictionary to store the emmision probabilities - stores the count of the occurance of each word given a POS
        count_W_given_S = {tag: {} for tag in set(pos)}

        for x in range(len(data)):
            # Tags - Parts of Speech
            S = data[x][1]
            # Words - Individual words in the sentences after removing the POS tags
            W = data[x][0]

            # Count occurrences
            for s, w in zip(S, W):
                count_W_given_S[s][w] = count_W_given_S[s].get(w, 0) + 1
                self.dict_pos[s] += 1

        total = copy.deepcopy(sum(self.dict_pos.values()))
        for pos in self.dict_pos:
            self.dict_pos[pos] = copy.deepcopy(self.dict_pos[pos]) / total

        # HMM Training

        # We maintain the list of all Parts of Speech
        pos = [
            "adj",
            "adv",
            "adp",
            "conj",
            "det",
            "noun",
            "num",
            "pron",
            "prt",
            "verb",
            "x",
            ".",
        ]

        # Initialising a dcitionary to store the Transition Probability going from one state to another
        self.transitionProb = copy.deepcopy(
            {tag: {other_tag: 0 for other_tag in pos} for tag in set(pos)}
        )

        # Initialising a dictionary to store the Emmision Probability
        self.emissionProb = {tag: {} for tag in set(pos)}

        # Initialising a dictionary to store the Initial Probabilities
        self.initial_prob = {}

        # Going through the data and storing the counts in the dictionary
        for x in range(len(data)):
            # Tags - Parts of Speech
            S = data[x][1]
            # Words - Individual words in the sentences after removing the POS tags
            W = data[x][0]

            self.initial_state = S[0]
            self.initial_prob[self.initial_state] = (
                self.initial_prob.get(self.initial_state, 0) + 1
            )

            for x in range(len(S) - 1):
                self.transitionProb[S[x]][S[x + 1]] += 1

            for x in range(len(W)):
                if S[x] not in self.emissionProb:
                    self.emissionProb[S[x]] = {}
                self.emissionProb[S[x]][W[x]] = self.emissionProb[S[x]].get(W[x], 0) + 1

        # Normalize initial probabilities - probability of a part of speech appearing at the start
        total_initial = sum(self.initial_prob.values())
        self.initial_prob = {
            state: count / total_initial for state, count in self.initial_prob.items()
        }

        # Transition Probability
        for tag in self.transitionProb:
            row_sum = sum(self.transitionProb[tag].values())
            if row_sum > 0:
                for other_tag in self.transitionProb[tag]:
                    self.transitionProb[tag][other_tag] /= row_sum

        # Emission Probability
        for tag in self.emissionProb:
            row_sum = sum(self.emissionProb[tag].values())
            if row_sum > 0:
                for other_tag in self.emissionProb[tag]:
                    self.emissionProb[tag][other_tag] /= row_sum

    # Functions for each algorithm.

    # For the simplified model, we just return the max of all emmision probabilties wrt the POS.
    def simplified(self, sentence):
        list = []
        for word in sentence:
            max_prob = 0.0
            max_tag = "noun"
            for i in self.emissionProb:
                if word in self.emissionProb[i]:
                    prob = self.emissionProb[i][word]
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = i
            list.append(max_tag)
        return list

    # HMM Algorithm
    def hmm_viterbi(self, sentence):
        pos = [
            "adj",
            "adv",
            "adp",
            "conj",
            "det",
            "noun",
            "num",
            "pron",
            "prt",
            "verb",
            "x",
            ".",
        ]

        # Sentence dictionary contains the probability and the path which leads to a particular hidden state
        words = []
        for word in sentence:
            words.append(word)
        sentence_dictionary = {word: {tag: {} for tag in set(pos)} for word in words}

        # Handling first word differently - as the first layer has to have Initial Probability and Emission Probability
        result = []

        for tag in pos:
            prob = 1
            max_prob = 0
            if words[0] in self.emissionProb[tag]:
                prob *= self.emissionProb[tag][words[0]] * self.initial_prob[tag]
            else:
                prob = self.small_emmision_prob * self.initial_prob[tag]
            sentence_dictionary[words[0]][tag] = (prob, [tag])
            if prob >= max_prob:
                max_prob = prob
                list = [tag]
        result = list

        # If the sentence contains only one word the result is returned
        if (len(words)) == 1:
            return result

        # Handling for the rest of the words (observed states) - by taking transmission probability, transmission probability and the probability from the previous state.
        result = []
        for x in range(1, len(words)):
            word = words[x]
            prev_word = words[x - 1]

            for tag_curr in pos:
                max_prob = 0
                list = []
                for tag_prev in pos:
                    prob = 1
                    if word in self.emissionProb[tag_curr]:
                        prob *= (
                            self.emissionProb[tag_curr][word]
                            * self.transitionProb[tag_prev][tag_curr]
                            * sentence_dictionary[prev_word][tag_prev][0]
                        )
                    # If word is not in the current POS , we consider a very small hard-coded value for the emmision probability
                    else:
                        prob *= (
                            self.small_emmision_prob
                            * self.transitionProb[tag_prev][tag_curr]
                            * sentence_dictionary[prev_word][tag_prev][0]
                        )

                    # We store only the max coming to a particular hidden state from the previous states
                    if prob >= max_prob:
                        max_prob = prob
                        list = copy.deepcopy(
                            sentence_dictionary[prev_word][tag_prev][1]
                        )

                list.append(tag_curr)
                sentence_dictionary[word][tag_curr] = (max_prob, list)

            result = copy.deepcopy(list)

        return result

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
