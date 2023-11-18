# BOTZ
TXST Human Factors Bot Detection 


-----------------------------installed components ---------------------
pip install -U scikit-learn






----------------------------notes for the project-----------------------
We need to determine the attribute that we want to learn on, and then pass the model examples that contain those attributes 

possibly need to containerize the csv file for better results when training 

csv file has benn put into a table for ease of reading. This is done by accessing the command palette (ctrl + shift + p) and (having already installed the 'CSV to Table' extension) searching for 'Convert to table from CSV(.csv / commas)' while on the .csv file you wish to convert

convert to numerical values (convert locations to gps coordinates or area codes / ASCII values) 


-------------------------For Daniel--------------------------------------
going over knn and determine how best to containerize the data and implement into the model

look into tensor flow and get the beginning of the paper ready for adding the material that we are finding 
-------------------------For Omar----------------------------------------
learn sklearn and tensorflow

-------------------------November 18th meeting---------------------------
Apply a constraint on the program, about 10,000 or less. That is 5x less than the dataset we are working with. The application of our project is work within smaller groups and companies

Group size, test size, how we get new data, relation between new data and current results

Total data set{x}; Evaluation data set{E disjunction of X}; Human data set{H union of X}
    E = evaluation data set; X = total data set; H = human data set

Testing:
    batching: number of pieces of data that we are feeding into the algorithm at a time
        if batching = 128 and data set = 50,000, then (data set / batching) = Batches
    epochs: the number of times the total number of batches that are run through the algorithm
        E - 1 (rand(x))

    *have to determine the best amount for the batch and epoch amount. Could maybe run it within a loop to help determine it, but that might take awhile to complete

Daniel managed to get the program running, obtaining an accuracy of 50%. This is a big milestone in our program. We are attempting to achieve around a 95% accuracy rate
    update: 2:47pm - we are now having some slight issue trying to get the program to run again
            3:04pm - discovered that a singular backslash in the dataset was responsible for the issue
            3:19pm - an epoch of 10, and batch size of 16 gave a 60% accuracy

A new .table file was made by Omar, printing the updated 'modded_detection_data' into a table format
