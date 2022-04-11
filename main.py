# Assignment 2 AI & ML
# Ken Carr

# import the wikipedia library for research
import wikipedia
import sumy
import numpy as np
from sklearn import tree
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from wikipedia.wikipedia import summary
from sklearn.datasets import load_iris

# FUNTIONS
# funtion to summarize research with a target number of sentances
def summarize(research, num_sentences):
    parser = PlaintextParser.from_string(research, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    research_summary = summarizer(parser.document, num_sentences)
    # return the list of sentances in summary as a concatinated paragraph
    return ' '.join(str(e) for e in research_summary)

def save_research(topics_list): # takes a list of topics in and does research that is saves into a single file
    output = open('research.txt','a') # if file dosen't exit, it creates it
    for x in range(len(topics_list)): # outputs the wikipedia search of each topic to a file
        temp = wikipedia.summary(topics_list[x])
        output.write(temp + "\n\n")
#        print(topics_list[x])
#        print(temp)
    output.close()

def create_summary_file(num_sentences):
    research_file = open('research.txt','r') # read only access of research data to be summarized
    research_data = research_file.read()
    research_file.close()
    summary_file = open('research_summary.txt','w') # rewrites summary based on new user requested length
    summary_file.write(summarize(research_data, num_sentences))
    summary_file.close()

print("\n*** Your AI Assitant ***\n")

run = True
while bool(run) == True:
    choice = input("\n\nSelect what AI can help you with:\n1 - Research\n2 - Check For Attacks\n3 - Flower Identifier\nE - EXIT\n")
    if choice == '1':
        print("\n=== Research Assistant ===\n")
        topic_list = list() # empty list started
        num_topics = int(input("\nHow many topics do you have to research (0 if you want to resumarize previous search)?   "))
        for x in range(num_topics): # put all topics to research in a list
            topic = input("Enter topic " + str(x+1) + " for research: ")
            topic_list.append(topic)
        save_research(topic_list)
        num_sentences = int(input("\nHow many sentences would you like your research to be? "))
        # funtion takes research and summarizes it to user defined size
        create_summary_file(num_sentences)
        print("\nI can help you with that research!")
        file = open('research_summary.txt')
        summary_from_file = file.read()
        file.close() #close stream
        print("\n***** Here is what my Aritifically Intellegent reseach found: *****\n" + summary_from_file + "\n\n")
    elif choice == '2':
        print("\n=== Air Defense Check ===")
        # Training Data
        features = [[42,60],[35,55],[29,35],[23,24],[26,28],[22,25]]
        # Training Markers: 1 = bomber & 0 = fighter
        plane_identifier = [1,1,1,0,0,0]
        # Create a decision tree classifier to train on the data
        tree_classifier = tree.DecisionTreeClassifier()
        tree_classifier = tree_classifier.fit(features,plane_identifier)
        # data to process to draw prediction
        wingspan = input("\nWhat is the Wingspan in FT? ")
        fuselage = input("\nWhat is the Fuselage length in FT? ")
        # make and feedback prediction to user
        prediction = tree_classifier.predict([[wingspan,fuselage]])
        # turn number clasification into user understood feedback
        if prediction == 1:
            plane = "BOMBER"
        else:
            plane = "FIGHTER"
        print("\n*** A " + plane + " IS COMMING! ***\n")
    elif choice == '3':
        iris_data_set = load_iris() # load iris dataset to work with
        # 3 values removed for testing trained model
        test_idx = [0,50,100]
        train_target = np.delete(iris_data_set.target, test_idx)
        train_data = np.delete(iris_data_set.data, test_idx, axis=0)
        # training data minus test data for validation of trainned model funtionality            test_target = iris_data_set.target[test_idx]
        test_target = iris_data_set.target[test_idx]
        test_data = iris_data_set.data[test_idx]
        # train the classifier that can distinguish the different types of flowers
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_data, train_target)
        print("\n***** Flower Finder *****")
        print("\n1 - Validate training model: ")
        print("2 - Use AI to Determine Flower Type!")
        selection = input("\nPick a Action: ")
        if selection == '1':
            test_count=0
            test_prediction = clf.predict(test_data) # predictions from trained model
            # each solution of the output of the predicted answer from the trainned model is compared to a known answer
            # this is a self checking funtion like a automated calibration step in a system to validate proper funtion
            if test_target[0] == test_prediction[0]:
                print("\nMsetosa Trainning Model PASSED")
                test_count = test_count + 1
            if test_target[1] == test_prediction[1]:
                print("\nVersicolor Trainning Model PASSED")
                test_count = test_count + 1
            if test_target[2] == test_prediction[2]:
                print("\nVirginica Trainning Model PASSED")
                test_count = test_count + 1
            if test_count == 3:
                print("\n##### Model Is Trainned Accurately! #####\n")
            else:
                print("\n!!!!! Trainned Model Failed !!!!!")
        elif selection == '2':
            # user values to be ran through trainned model
            septal_length = float(input('Enter the septal length: '))
            septal_width = float(input('Enter the septal width: '))
            petal_length = float(input('Enter the petal length: '))
            petal_width = float(input('Enter the petal width: '))
            # user input must be a 2 dimentional array before being entered into predict funtion
            user_flower = [[septal_length,septal_width,petal_length,petal_width]]
            ans = clf.predict(user_flower)
            if ans == 0:
                print("\n*** Your Flower is A Msetosa ***")
            elif ans == 1:
                print("\n*** Your Flower is A Versicolor ***")
            elif ans == 2:
                print("\n*** Your Flower is A Virginica ***")
            else:
                print("Error - Not Classified")
        else:
            print("\nMake a valid selection")
    elif choice.lower() == 'e': # accounts for upper and lower case e
        print("\n*** Goodbye ***\n")
        run = False
    else:
        print("*** Not A Valid Selection ***\n*** Try Again ***\n")


