import os
import time
import subprocess

def git_push(interval_minutes, commit_message=None):
    while True:
        # Add all changes
        subprocess.run(['git', 'add', '.'])

        # Commit with the provided message
        # set commit message to automated commit if no message is provided
        if not commit_message:
            commit_message = "Automated commit"
        subprocess.run(['git', 'commit', '-m', commit_message])

        # Push to the remote repository
        subprocess.run(['git', 'push'])

        # Pull the latest changes
        # subprocess.run(['git', 'pull'])

        print(f"Changes pushed to the repository. Waiting for {interval_minutes} minutes before the next push...")
        
        # Wait for the specified interval
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    # Set the interval in minutes
    interval = 10  # Example: 10 minutes

    # Set your commit message
    # commit_msg = "Automated commit"

    # Start the git push loop
    git_push(interval)
