## How to run the script
To set up the environment, run:
```bash
source setup.sh
```

Some example commands:
```python
python3 wmdp.py --mode "base" --task "wmdp-bio"
python3 wmdp.py --mode "unlearned" --task "wmdp-bio"

python3 wmdp.py --mode "base" --task "wmdp-chem"
python3 wmdp.py --mode "unlearned" --task "wmdp-chem"

python3 wmdp.py --mode "base" --task "wmdp-cyber"
python3 wmdp.py --mode "unlearned" --task "wmdp-cyber"
```

To look at the eval logs:
```bash
inspect view --log-dir "./log"
```