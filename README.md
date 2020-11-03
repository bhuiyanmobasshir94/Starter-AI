# Starter-AI
## Step 1: configuration files
Add these files to the root of your repository.
- setup.cfg — configuration for flake8 and mypy.
- pyproject.toml — configuration for black.

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
