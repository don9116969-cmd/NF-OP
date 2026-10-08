"""
Automated Git Synchronization Engine for Multi-Index Option Trading Bot.
Automatically commits and pushes active position updates and journal logs to GitHub.
"""
import os
import subprocess
import time

def git_sync_push(commit_msg: str, max_retries: int = 2) -> bool:
    """
    Stages and pushes journal, active positions, and ledger to GitHub remote origin.
    Non-blocking: Uses non-interactive git environment and strict timeouts to ensure
    it never halts trading bot execution.
    """
    try:
        # Check if inside git repository
        repo_check = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, timeout=5)
        if repo_check.returncode != 0:
            return False

        # Set user if not already set (e.g. inside GitHub Actions runner)
        subprocess.run(["git", "config", "user.name", "Nifty Bot Cloud Runner"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        subprocess.run(["git", "config", "user.email", "bot@github-actions.internal"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)

        # Stage journal, candle datasets, and execution state tracking files
        files_to_add = [
            "data/trade_journal.csv",
            "data/trade_journal.md",
            "data/active_positions.json",
            "data/sent_alerts.json",
            "data/live_paper_trades.csv",
            "data/paper_trading_ledger.csv",
            "data/audited_days.json",
            "data/nifty_1min_real.csv",
            "data/banknifty_1min_real.csv"
        ]
        existing_files = [f for f in files_to_add if os.path.exists(f)]
        if not existing_files:
            return False

        subprocess.run(["git", "add"] + existing_files, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)

        # Check if there are changes staged
        diff_res = subprocess.run(["git", "diff", "--staged", "--quiet"], check=False, timeout=5)
        if diff_res.returncode == 0:
            # No changes to commit
            return True

        # Commit changes locally
        subprocess.run(["git", "commit", "-m", f"{commit_msg} [skip ci]"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)

        # Determine environment and timeouts
        is_ci = bool(os.getenv("GITHUB_ACTIONS"))
        retries = max_retries if is_ci else 1
        timeout_sec = 15 if is_ci else 6

        git_env = os.environ.copy()
        git_env["GIT_TERMINAL_PROMPT"] = "0"

        # Check for GitHub Actions environment with GITHUB_TOKEN
        remote_target = "origin"
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPOSITORY")
        if token and repo:
            remote_target = f"https://x-access-token:{token}@github.com/{repo}.git"

        # Push with pull --rebase retry loop
        for attempt in range(retries):
            try:
                subprocess.run(
                    ["git", "pull", "--rebase", "--autostash", remote_target, "main"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=git_env,
                    timeout=timeout_sec
                )
                push_res = subprocess.run(
                    ["git", "push", remote_target, "main"],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=git_env,
                    timeout=timeout_sec
                )
                if push_res.returncode == 0:
                    print(f"[GIT-SYNC] Successfully pushed to GitHub: '{commit_msg}'")
                    return True
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1)

        print("[GIT-SYNC] Local commit recorded. Remote push deferred.")
        return False
    except Exception as e:
        print(f"[GIT-SYNC] Auto-sync notice: {e}")
        return False
