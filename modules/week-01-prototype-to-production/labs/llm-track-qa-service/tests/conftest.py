import os

# Deterministic tests: never let an ambient real-looking OPENAI_API_KEY in the
# developer's shell environment cause tests to make real, billed network calls
# to api.openai.com. This must run at module level (not inside a fixture) so
# it takes effect before pytest's collection phase imports app.main, which
# instantiates Settings() at import time.
os.environ["OPENAI_API_KEY"] = ""
