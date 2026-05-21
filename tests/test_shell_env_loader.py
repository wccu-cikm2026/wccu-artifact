import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ShellEnvLoaderTests(unittest.TestCase):
    def test_shell_helper_loads_dotenv_without_overriding_existing_env(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / '.env'
            env_path.write_text(
                'GEMINI_API_KEY=from_file\n'
                'GEMINI_AGENT_MODEL=gemini-3.1-flash-lite\n'
                'WCCU_COOPER_SEED=7\n',
                encoding='utf-8',
            )
            env = os.environ.copy()
            env['WCCU_ENV_FILE'] = str(env_path)
            env['GEMINI_API_KEY'] = 'from_shell'
            cmd = [
                'bash',
                '-lc',
                'source scripts/lib/load_wccu_env.sh; '
                'printf "%s|%s|%s" "$GEMINI_API_KEY" "$GEMINI_AGENT_MODEL" "$WCCU_COOPER_SEED"',
            ]
            out = subprocess.check_output(cmd, cwd=root, env=env, text=True)
            self.assertEqual(out, 'from_shell|gemini-3.1-flash-lite|7')


if __name__ == '__main__':
    unittest.main()
