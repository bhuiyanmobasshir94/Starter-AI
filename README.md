# Starter-AI
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Step 1: configuration files
Add these files to the root of your repository.
- [setup.cfg](https://github.com/bhuiyanmobasshir94/Starter-AI/blob/main/setup.cfg) — configuration for flake8 and mypy.
- [pyproject.toml](https://github.com/bhuiyanmobasshir94/Starter-AI/blob/main/pyproject.toml) — configuration for black.

## Step 2: requirements
Install the required libraries with
```
pip install black flake8 mypy
```

## Step 3: black
There are 100500 ways to format the code. Formatters like black or yapf modify the code to satisfy a pre-defined set of rules.
It is easier to read codebase that has some standards. When you work on the code for hours and need to switch a context between different coding styles, it drains “willpower energy” — no need to do it without a good reason.
Running
```
black .
```
will reformat all python files to follow the set of rules by black.

## Step 4: flake8
Running
```
flake8
```
will not modify the code, but will check code for syntax issues and output them to the screen.
Fix them.

## Step 5: mypy
Python does not have mandatory static typization, but it is recommended to add types to the function arguments and return types.
For example:
```
class MyModel(nn.Module):
    ....
def forward(x: torch.Tensor) -> torch.Tensor:
    ....
    return self.final(x)
```
You should add typing to the code. It makes it easier to read the code.
You can use the mypy package to check arguments and function types for consistency.
After updating the code run mypy on the whole repo:
```
mypy .
```
If mypy found issues — fix them.

## Step 6: pre-commit hook
Running **flake8**, **black**, **mypy** manually all the time is annoying.
There is a tool called **[pre-commit hook](https://pre-commit.com/)** that addresses the issue.
To enable it — copy this file to your repo: *[.pre-commit-config.yaml](https://github.com/bhuiyanmobasshir94/Starter-AI/blob/main/.pre-commit-config.yaml).*
You need to install the pre-commit package on your machine with:
```
pip install pre-commit
```
And initialize with:
```
pre-commit install
```
You are good to go.
From now on, on every commit, it will run a set of checks and not allow the commit to pass if something is wrong.
The main difference between the manual running of the black, flake8, mypy is that it does not beg you to fix issues, but forces you to do this. Hence, there is no waste of “willpower energy.”

To check all with **pre-commit** run the following:
```
pre-commit run --all-files
```

## Step 7: Github Actions
You added checks to the pre-commit hook, and you run them locally. But you need a second line of defense. You need Github to run these checks on every pull request. Way to do it is to add file *[.github/workflows/ci.yaml](https://github.com/bhuiyanmobasshir94/Starter-AI/blob/main/.github/workflows/ci.yml)* to the repo.
There are lines:
```
- name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install black flake8 mypy
    - name: Run black
      run:
        black --check .
    - name: Run flake8
      run: flake8
    - name: Run Mypy
      run: mypy retinaface
```
that tell GitHub what to check. I also recommend to give up the practice of pushing the code directly to the master branch. Create a new branch, modify the code, commit, push to Github, create a pull request, and merge to master. It is a standard in the industry, but it is exceptionally uncommon in the academy and among Kagglers. If you are not familiar with these tools, it may take more than 20 minutes to add them and fix errors and warnings.
Remember this time. In the next project, add these checks in the first commit, when no code is written. From that moment, every small commit will be checked, and you will need to fix at most a couple lines of code every time: tiny overhead, excellent habit.

> If following two error persit
```
 FileNotFoundError: [Errno 2] No such file or directory: 'c:\\programdata\\anaconda3\\Lib\\venv\\scripts\\nt\\python.exe'

 FileNotFoundError: [Errno 2] No such file or directory: 'c:\\programdata\\anaconda3\\Lib\\venv\\scripts\\nt\\pythonw.exe'
```
Just copy the files there. This is just for anaconda environment.

Resource: [Building and testing Python](https://docs.github.com/en/free-pro-team@latest/actions/guides/building-and-testing-python)

If this happen - pre-commit fails when setting up the black environment
Do this `pre-commit install --install-hooks`
Ref: [1](https://github.com/psf/black/issues/1180#issuecomment-565624865)

Credit:  [Vladimir Iglovikov](https://medium.com/kaggle-blog/i-trained-a-model-what-is-next-d1ba1c560e26)
