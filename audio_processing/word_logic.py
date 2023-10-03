import numpy as np


# function for valid words for tick tack toe game
# returns combined string of valid words
# valid words are: a, b, c, 1, 2, 3, stopp, rex and combinations are: a1, a2, a3, b1, b2, b3, c1, c2, c3

class WordLogic:
    def __init__(self):
        self.valid_words = ["a", "b", "c", "1", "2", "3", "stopp", "rex"]
        self.valid_combinations = ["a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3"]
        self.rex = False
        self.current_word = ""
        self.current_combination = ""
    
    # function for valid words for tick tack toe game, gets prediction from model as a string, checks if valid word, and stores it in a combination_word
    # waits for second word, checks if valid word, and stores it in a combination_word
    # as soon as combination_word gets 2 valid words, it returns the combination_word
    # if no valid word, it returns "other"
    # The word rex is the word that initializes the process of getting a combination_word
    # if the word rex is said, the function returns "rex" and waits for the first valid word and stores it in a combination_word and waits for the second valid word and stores it in a combination_word

    def command(self, prediction):
        if prediction == "stopp":
            print("Game stopped")
            self.current_combination = "stopp"
            return "stopp"
        if prediction == "rex":
            self.rex = True
            print("Rex activated")
            return "rex"
        elif self.rex == True:
            if prediction in self.valid_words:
                self.current_word = self.current_word + prediction
                print(f"Current word: {self.current_word}")
                if self.current_word in self.valid_combinations:
                    self.current_combination = self.current_word
                    print(f"Current combination: {self.current_combination}")
                    self.current_word = ""
                    self.rex = False
                    return self.current_combination
            else:
                print("No valid word")
                self.current_word = ""
                self.rex = False
        else:
            print("No valid word")
            self.current_word = ""
            self.rex = False
        
    def get_combination(self):
        return self.current_combination
    
    def reset_combination(self):
        self.current_combination = ""
    
    


    




    
    