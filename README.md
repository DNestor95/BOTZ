# BOTZ
TXST Human Factors Bot Detection 


------------------------INSTALLED COMPONENTS---------------------

pip install -U scikit-learn    
pip install -U tensorflow     
pip install -U numpy


------------------------TO RUN-----------------------------------

HELLO!     
If you are attempting to run this program or want to use this for your own testing please make sure that you have the modeified data set downloaded and not the original. You can also replace the dataset if need be at your own risk as modifications have been made to this data for the program to run correct 

Linear regression model testing command: 
python3 newTFTest.py

Classification model testing command: 
python3 classificationTF.py

The above commands will run each of the models and give you the output and accurace in relation to the evaluation file that is included. 

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

Both the linear regression model and the classification model are created. classification is still having issues running but I am working on it and will hopefully have a working model to add to the testing for the project. 
