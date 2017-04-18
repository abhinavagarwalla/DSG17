Add all the plots, and exploratory data analysis stuff in eda.py
Global configuration in configs.py

Step 1, generate_settings.py
For feature-level processing data, use feature_generator.
Then, generate the folds for all your experiments using splitter.py (currently not required)
To make more features, just add functions to this file, and add the resulting npy file name in generate_datasets.py

Then generate_dataset, to form the data.
