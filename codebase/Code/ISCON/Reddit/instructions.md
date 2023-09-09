### Instructions for running the ISCON model

1. Download the dataset as per the instructions.
2. For the LastFM dataset.
3. For the Reddit dataset -
    1. Run the file named 'reddit_script.py' with the data file in the same directory as the above mentioned file.
    2. Copy the output file to the folder named data and rename to 'reddit.tsv'.
    3. Run the file named 'main.py' in the folder 'src' with the flags -> --data_name reddit --output results_reddit --contexts 20 --topk 3.
    4. The results would be in the folder 'output'.