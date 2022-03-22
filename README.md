### EEGEyeNet Classification / Regression Using Riemanian Methods

Files in this repo are split up between classification / regression task.

## Setup (DO THIS FOR EITHER TASK)

1. Install all packages in requirements.txt by running
   `pip3 install -r requirements.txt`

## CLASSIFICATION TASK REPRODUCTION STEPS

1. Download the zipped data from this link: https://drive.google.com/file/d/1TMpNu_hQVGZqOlziCUBbJCXA8e6ER0wD/view?usp=sharing
2. Unzipping the file should yield a folder named `prepared_data/` containing multiple files ending with `.npz`.
3. Drag the `prepared_data/` folder into the `data/` folder in this repo.
4. After installing all dependencies, run `main_classification.py`. Results are both printed to stdout and saved in `classification_results.csv`.

## REGRESSION TASK REPRODUCTION STEPS

1. Download the Position_task_with_dots_synchronised_min.npz (https://osf.io/ge87t/) from OSFHome (https://osf.io/ktv7m/). Move this file directly into the `data/` folder.
2. For the regression tasks, repro using regression_notebook.ipynb.
3. To open notebooks, type "jupyter-notebook" (without the quotations) in your terminal, then copy paste the given urls into a browser.
4. Click "Run All" for both notebooks in your Jupyter notebook UI.
5. The notebook may take 30 minutes to an hour to run
