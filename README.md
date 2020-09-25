# Vehicle type recognition from image data

This project aims to recognize vehicles through images. It is based on a Convolutional Neural Network (from now on it is called CNN) following the essence of an unsupervised Machine Learning model.

The intention is to classify a dataset composed of images that includes different types of vehicles including cars, bicycles, boats, trucks, etc. with a total of 17 categories.

The model consists of training data with the class labels and test data without the labels. The project deals to predict the secret labels for the test data.

The dataset has been collected from the Open Images dataset (over 9 million images) using a subset, selected to contain only vehicle categories among the total of 600 object classes.

The dataset consists of two files listed below.

    - The training set (train.zip): a set of images with true labels in the folder names. The zip file contains altogether 28045 files organized in folders. The folder name is the true class; i.e., "Boat" folder has all boat images, "Car" folder has all the car images and so on.
    - The test set (test.zip): a set of images without labels. The zip file contains altogether 7958 files in a single folder. The file name is the id for the solution's first column; i.e., the predicted class for file "000000.jpg".

![Image](https://cdn.pixabay.com/photo/2016/11/29/10/01/vw-bully-1868890_960_720.jpg)

---

## **Formatting**
Your readers will most likely view your README in a browser so please keep that in mind when formatting its content: 
- Use proper format when necesary (e.g.: `import pandas as pd`). 
- Categorize content using two or three levels of header beneath. 
- Make use of **emphasis** to call out important words. 
- Link to project pages for related libraries you mention. Link to Wikipedia, Wiktionary, even Urban Dictionary definitions for words of which a reader may not be familiar. Make amusing cultural references. 
- Add links to related projects or services. 

> Here you have a markdown cheatsheet [Link](https://commonmark.org/help/) and tutorial [Link](https://commonmark.org/help/tutorial/).


## **Start writing ASAP:**
*Last but not least, by writing your README soon you give yourself some pretty significant advantages. Most importantly, you’re giving yourself a chance to think through the project without the overhead of having to change code every time you change your mind about how something should be organized or what should be included.*


## **Suggested Structure:**

### :raising_hand: **Name** 
Self-explanatory names are best. If the name sounds too vague or unrelated, it may be a signal to move on. It also must be catchy. Images, Logo, Gif or some color is strongly recommended.

### :baby: **Status**
Alpha, Beta, 1.1, Ironhack Data Analytics Final Project, etc... It's OK to write a sentence, too. The goal is to let interested people know where this project is at.

### :running: **One-liner**
Having a one-liner that describes the pipeline/api/app is useful for getting an idea of what your code does in slightly greater detail. 

### :computer: **Technology stack**
Python, Pandas, Scipy, Scikit-learn, etc. Indicate the technological nature of the software, including primary programming language(s), main libraries and whether the software is intended as standalone or as a module in a framework or other ecosystem.

### :boom: **Core technical concepts and inspiration**
Why does it exist? Frame your project for the potential user. Compare/contrast your project with other, similar projects so the user knows how it is different from those projects. Highlight the technical concepts that your project demonstrates or supports. Keep it very brief.

### :wrench: **Configuration**
Requeriments, prerequisites, dependencies, installation instructions.

### :see_no_evil: **Usage**
Parameters, return values, known issues, thrown errors.

### :file_folder: **Folder structure**
```
└── project
    ├── __trash__
    ├── .gitignore
    ├── .env
    ├── requeriments.txt
    ├── README.md
    ├── main_script.py
    ├── notebooks
    │   ├── notebook1.ipynb
    │   └── notebook2.ipynb
    ├── package1
    │   ├── module1.py
    │   └── module2.py
    └── data
        ├── raw
        ├── processed
        └── results
```

> Do not forget to include `__trash__` and `.env` in `.gitignore` 

### :shit: **ToDo**
Next steps, features planned, known bugs (shortlist).

### :information_source: **Further info**
Credits, alternatives, references, license.

### :love_letter: **Contact info**
Getting help, getting involved, hire me please.

---

> Here you have some repo examples:
- [Mamba (OCR-Translator-Assistant)](https://github.com/YonatanRA/OCR-translator-assistant-project)
- [Art Classification](https://github.com/serguma/art_classification)
- [OSNet-IBN (width x 1.0) Lite](https://github.com/RodMech/OSNet-IBN1-Lite)
- [Movie Founder](https://github.com/Alfagu/final-project-Ironhack-0419mad)
- [Convolutional Neural Network to detect Pneumonia](https://github.com/jmolins89/final-project)
- [Brain tumor detection project](https://github.com/alonsopdani/brain-tumor-detection-project)
- [Policy-Gradient-Methods](https://github.com/cyoon1729/Policy-Gradient-Methods)

> Here you have some tools and references:
- [Make a README](https://www.makeareadme.com/)
- [Awesome README](https://github.com/matiassingers/awesome-readme)
- [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

