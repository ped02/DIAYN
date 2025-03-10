# Diversity Is All You Need (CS8803 DRL)

## Setup
```
git clone <this repo> <name of directory to clone into>
cd <name of directory to clone into>
./scripts/setup.sh # or .\scripts\setup.bat for window
```

The `setup.sh` or `setup.bat` script will setup a virtual environment for the project.

Note: If you're cloning into a workspace, I recommend changing name of directory to clone into. VS Code may have problem with code suggestions.

```
# Activate venv

# Install Module
python3 -m pip install -e .
```

## Note on project organization
Not very strict.

(Rachanon) put running code and experiments into the `examples` directory. The code in the `DIAYN` module shouldnt really change across experiments unless you're adding a new feature.
