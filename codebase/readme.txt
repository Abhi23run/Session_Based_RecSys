––––––––––––––––
CSE6240 Project: Session Based Recommender System
––––––––––––––––


Hello! Thank you for reviewing our project. In our project we have explored two premier session-based recommendation systems – Heterogenous Global Graph Neural Networks and Implicit Session Contexts – and applied them to two separate datasets, the LastFM and Reddit datasets. These are both popular datasets to be used for session-based recommendation systems, and were easy to use with our two implementations. 


––––––––––––––––
Getting the data
––––––––––––––––

Both datasets which we used were too large to include here, so they can be downloaded from the following links:


LastFM Raw - http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

Reddit Raw - https://www.kaggle.com/datasets/colemaclean/subreddit-interactions

LastFM Cleaned - https://drive.google.com/drive/folders/1wgw3G6dBcJ_HQ10D4Kd5bggWmzXnTWd_?usp=share_link

Once the data is downloaded, you will be good to go! Just make sure you add the data to the correct folder as labeled in the instructions. 


**Please note that hte LastFM Raw and LastFM Cleaned datasets are the exact same data. The LastFM Cleaned dataset above is just a cleaned version of the LastFM Raw dataset as completed by the HG-GNN data cleaning process which is baked into the model running. Since the code for this is slightly different in the ISCON implementation, a cleaned version of the data was provided for ease of use in the LastFM ISCON model.**



––––––––––––––––
Running the Code
––––––––––––––––

Our code has been split by method and dataset, leading to 4 separate code chunks for running. These are provided in the parent directory "Code" in this folder. The folder "Code" is then split into "ISCON" and "HG-GNN," which are both further divided by datasets "LastFM" and "Reddit". Inside each of these folders named after the data you will find the code used to generate the results for that model on that dataset. Steps we took to implement this code will be explained below.


We utilized code for these two approaches from the HG-GNN and ISCON implementation Github repositories, which are found here https://github.com/0215Arthur/HG-GNN and here https://github.com/claws-lab/iscon respectively. 



In order to run each model, one should open a terminal window and tab to the respective folder which one hopes to run. Then, they should move the respective data for said folder into the 



ISCON - LastFM
–––––––––––––

The following steps must be completed in order to run this code

1. Open a terminal window and tab over to this directory using "cd."

2. Download the LastFM CLEAN dataset and insta

3. Install the requirements using the following line of code: pip install -r requirements.txt

4. Run the model using the following command: sh demo.sh 

Please note that the demo.sh file contains the running instructions for this code. This can be changed in the command structure in demo.sh by adding more "--<parameter>" commands. 

Additionally, an additional way to generate and contextualize sessions was attempted on this dataset. This approach was commented out of the submitted folder, but by uncommenting this approach. If you would like to explore this additional implementation, you can comment out the current function "session_find_and_split" in main.py and uncomment the commented out function. In our experimentation, the currently uncommented implementation performed slightly better. 



ISCON - Reddit
–––––––––––––

The following steps must be completed in order to run this code:

1. Download the Reddit - RAW dataset from the above link and move it into the data file. 

2. Run the reddit_script.py file in the data folder. Make sure the that output ('reddit.tsv') remains in this folder

3. Run the file named "main.py" in the folder "src" with the following flags: --data_name reddit --output results_reddit --contexts 20 --topk 3

4. The results will be returned in the folder "output."


HG-GNN - LastFM
–––––––––––––
The following steps must be completed in order to run this code:

1. Download the LastFM - RAW datafile listed above and move it to the raw_dataset folder

2. Install all the requirements using the following line of code: pip install -r requirements.txt

3. Run the main.py file in this directory



HG-GNN - Reddit
–––––––––––––
The following steps must be completed in order to run this code:

1. Download the Reddit - RAW datafile listed above and move it to the raw_dataset folder

2. Install all the requirements using the following line of code: pip install -r requirements.txt

3. Run the main.py file in this directory




