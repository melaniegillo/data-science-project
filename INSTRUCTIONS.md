# Final Project Workshop

**Debugging • Project Structure • Best Practices**

---

# Project

- GitHub repository
- 10-page technical report
- Clean, documented code

---

# Part 1: Debugging

---

# Types of Bugs

**1. Syntax Errors** - Python can't parse your code
```python
if x > 5  # Missing colon
```

**2. Runtime Errors** - Crashes during execution
```python
result = 10 / 0  # ZeroDivisionError
```

**3. Logic Errors** - Code runs but produces wrong results
```python
def calculate_discount(price, percent):
    return price + (price * percent)  # Should be minus!
```

---

# Reading Error Messages

```python
Traceback (most recent call last):
  File "script.py", line 15, in <module>
    result = divide(10, 0)
  File "script.py", line 3, in divide
    return a / b
ZeroDivisionError: division by zero
```

**How to read:**
1. Start from **bottom** - that's the actual error
2. Work **up** to see the call chain
3. Look for **your code** (not library code)
4. **Line numbers** tell you exactly where

---

# Exception Handling

**Without handling** (program crashes):
```python
def divide(a, b):
    return a / b

result = divide(10, 0)  # 💥 Crash!
```

**With handling** (program continues):
```python
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None

result = divide(10, 0)  # Handles gracefully
print("Program continues")
```

---

# Multiple Exception Handling

```python
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Division by zero")
        return None
    except TypeError:
        print(f"Error: Cannot divide {type(a)} by {type(b)}")
        return None
    finally:
        print("Division operation completed")  # Always runs
```

**Best Practice:** Catch specific exceptions, not all!

---

# Part 2: Modern Project Structure

---

# Why Project Structure Matters

**Bad:** One giant `.py` file
```
my_project.py  # 2000 lines of code
```

**Good:** Organized structure
```
my-project/
├── README.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   └── main.py
└── tests/
    └── test_main.py
```

**Benefits:** Understandable • Maintainable • Importable

---

# Standard Project Layout

```
my-research-project/
├── .gitignore
├── README.md
├── pyproject.toml      # Project configuration
├── src/                # Source code
│   ├── __init__.py
│   ├── main.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
└── tests/              # Tests
    └── test_main.py
```

**Key:** `__init__.py` makes directories into importable packages

---

# The `pyproject.toml` File

Modern Python configuration (replaces `setup.py`, `requirements.txt`):

```toml
[project]
name = "my-research-project"
version = "0.1.0"
description = "A short description"

[project.dependencies]
numpy = ">=1.24.0"
pandas = ">=2.0.0"
matplotlib = ">=3.7.0"
```

**This is the source of truth for your project!**

---

# Managing Dependencies with `uv`

**`uv`** = Fast, all-in-one tool (package installer + environment manager)

```bash
# Initialize new project
uv init

# Add dependencies
uv add numpy pandas matplotlib

# Install everything (from pyproject.toml)
uv sync

# Run code in environment
uv run python src/main.py
```

**No more:** `pip install`, `venv`, `requirements.txt` juggling!

---

# Code Quality with `ruff`

**`ruff`** = Fast linter + formatter

```bash
# Add as dev dependency
uv add --dev ruff

# Find problems
uv run ruff check .

# Auto-format code
uv run ruff format .
```

**What it does:**
- Finds bugs and style issues
- Enforces consistent formatting
- Catches common mistakes

---

# Your Project Checklist

- [ ] **Structured layout** (not one giant file)
- [ ] **README.md** with setup instructions
- [ ] **pyproject.toml** with dependencies
- [ ] **Exception handling** for file I/O, user input
- [ ] **Comments** explaining complex logic
- [ ] **Clean git history** (not 1 massive commit)
- [ ] **Tests** (at least basic ones)
- [ ] **AI_LOG.md** documenting AI usage

---

# Common Project Mistakes

**1. One Giant File**
- Split into modules!

**2. Hardcoded Paths**
```python
# Bad
data = pd.read_csv('/Users/alice/Desktop/data.csv')

# Good
from pathlib import Path
data_path = Path(__file__).parent / 'data' / 'data.csv'
data = pd.read_csv(data_path)
```

**3. No Error Handling**
- Files might not exist!
- Users might enter bad data!

---

# Common Project Mistakes (cont.)

**4. Unclear Variable Names**
```python
# Bad
d = [1, 2, 3]
x = sum(d) / len(d)

# Good
data_points = [1, 2, 3]
average = sum(data_points) / len(data_points)
```

**5. No Documentation**
- Future you won't remember what this does!
- Add docstrings and comments

**6. Not Using Version Control**
- Commit frequently with meaningful messages

---

# Part 3: Your Projects

---

# Project Requirements Reminder

**Code Requirements:**
- Python ≥3.10
- Clean, readable code
- Basic testing
- Good documentation
- Use of advanced features (OOP, file I/O, etc.)

**Report Requirements (10 pages):**
- Problem description
- Technical approach
- Implementation details
- Results and analysis
- AI usage log

---

# AI Usage Policy

**You CAN use:**
- Claude
- Any AI coding assistant

**You MUST:**
- Keep `AI_LOG.md` documenting what AI generated
- Understand and explain AI-generated code
- Test AI code (it's often wrong!)

**Rule:** If you can't explain it, you can't submit it

---

# Final Tips for Success

**1. Start Now**
- Don't wait until December 20!

**2. Commit Often**
- Git history shows your process

**3. Test As You Go**
- Don't write 500 lines then test

**4. Keep It Simple**
- Working simple code > broken complex code

---

# Resources

**Documentation:**
- Python docs: https://docs.python.org
- NumPy: https://numpy.org/doc
- Pandas: https://pandas.pydata.org/docs
- Matplotlib: https://matplotlib.org