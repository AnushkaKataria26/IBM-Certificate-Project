with open("src/serving/app.py", "r") as f:
    lines = f.readlines()

# find where logger is defined
log_idx = 0
for i, line in enumerate(lines):
    if line.startswith("logger = logging.getLogger(__name__)"):
        log_idx = i
        break

# we should move load_rag_index() below logging
