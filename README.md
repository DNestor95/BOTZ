# BOTZ
TXST Human Factors Bot Detection 
12/4/20023

------------------------INSTALLED COMPONENTS---------------------

pip install -U scikit-learn    
pip install -U tensorflow     
pip install -U numpy

<<<<<<< Updated upstream
=======
----------------------------------TO RUN--------------------------------
HELLO! 
if you are attempting to run this program or want to use this for your own testing please make sure that you have the modeified data set downloaded and not the original. You can also replace the dataset if need be at your own risk as modifications have been made to this data for the program to run correct 
>>>>>>> Stashed changes

------------------------TO RUN-----------------------------------

HELLO!     
If you are attempting to run this program or want to use this for your own testing please make sure that you have the modeified data set downloaded and not the original. You can also replace the dataset if need be at your own risk as modifications have been made to this data for the program to run correct 

Linear regression model testing command: 
python3 newTFTest.py

<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======

//classification function is completely broken and I cant get it to work anymore with the updated data set 
>>>>>>> Stashed changes
Classification model testing command: 
python3 classificationTF.py

The above commands will run each of the models and give you the output and accurace in relation to the evaluation file that is included. 
=======

!!!!!!NOTE!!!! THIS MODEL IS NOT WORKING THIS MODEL IS NOT PART OF THE TEST FOR THE SCOPE OF HUMAN FACTORS 
but you can try if you want :)
classification model testing command 
python3 classificationTF.py

the above commands will run each of the models and give you the output and accuracy in relation to the evaluation file that is included. 
>>>>>>> Stashed changes

Feel free to change the file and try your own


------------------------NOTES FOR THE PROJECT--------------------

We need to determine the attribute that we want to learn on, and then pass the model examples that contain those attributes 

Possibly need to containerize the csv file for better results when training 

The .csv file has benn put into a table for ease of reading. This is done by accessing the command palette (ctrl + shift + p) and (having already installed the 'CSV to Table' extension) searching for 'Convert to table from CSV(.csv / commas)' while on the .csv file you wish to convert

Convert to numerical values (convert locations to gps coordinates or area codes / ASCII values) 


------------------------FOR DANIEL-------------------------------

Going over knn and determine how best to containerize the data and implement into the model

Look into tensor flow and get the beginning of the paper ready for adding the material that we are finding 


------------------------FOR OMAR---------------------------------

Learn sklearn and tensorflow


------------------------NOVEMBER 18TH MEETING--------------------

Apply a constraint on the program, about 10,000 or less. That is 5x less than the dataset we are working with. The application of our project is work within smaller groups and companies

Group size, test size, how we get new data, relation between new data and current results

Total data set{x}; Evaluation data set{E disjunction of X}; Human data set{H union of X}
    E = evaluation data set; X = total data set; H = human data set

Testing:
    batching: number of pieces of data that we are feeding into the algorithm at a time
        if batching = 128 and data set = 50,000, then (data set / batching) = Batches
    epochs: the number of times the total number of batches that are run through the algorithm
        E - 1 (rand(x))

    Note: have to determine the best amount for the batch and epoch amount. Could maybe run it within a loop to help determine it, but that might take awhile to complete

Daniel managed to get the program running, obtaining an accuracy of 50%. This is a big milestone in our program. We are attempting to achieve around a 95% accuracy rate
    update: 2:47pm - we are now having some slight issue trying to get the program to run again
            3:04pm - discovered that a singular backslash in the dataset was responsible for the issue
            3:19pm - an epoch of 10, and batch size of 16 gave a 60% accuracy

A new .table file was made by Omar, printing the updated 'modded_detection_data' into a table format


------------------------NOVEMBER 23RD TESTING--------------------

Testing for linear regression done and accuracy of 91.25 is the best we are able to get with our current data set. This isnt exactly what we are looing for but it is actually alot better than we expect based on our private testing 


-------------------------November 23th testing---------------------------
<<<<<<< Updated upstream

Both the linear regression model and the classification model are created. classification is still having issues running but I am working on it and will hopefully have a working model to add to the testing for the project. 
=======
both the linear regression model and the classification model are created. classification is still having issues running but I am working on it and will hopefully have a working model to add to the testing for the project. 


-----------------Discussion on topic during dev---------------------------
This readme is going to talk about the steps that we are taking to make sure that the code we are
writng is acceptable and making sense as far as the scope of this project is concerned. 

Some of the main issues that we have had while attmepting this project are what the best use case for a product like this is going to be.

If we will be able to actually do some sort of machine learning to create a MVP and if that product can be tested and useable 

Can we process the data correctly to be able to find what we are looking for. Or that is get out machine learning tools pointed in the right direction to find what it is that we are looing for. 

During the setup of the project we looked at the way in which identifying a bot can be done and the ways that we would need to examine accounts to determine if they are a bot with the eye test. Things like the contents of the post and the interaction that they were having with other bots. Also the age of the account and how old each of the accounts are. All of these are something that a person would take into account when looking at a bot. But a good bot can beat those eye tests for some time and needs to be observed over time. 

Looking at data collected from twitter users we have lists of key words as well as hashtags and the date tweets are made of accounts that are confirmed to be either bots or not bots. Using a trend line and seperating the bots and people using tensorflow we were able to acheive over a 90% accuracy on the test results 
>>>>>>> Stashed changes
