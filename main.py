import subprocess
import sys

try:
    print("\nSTEP 1: Generating prompts/storyboard...\n")

    subprocess.run(
        [sys.executable, "prompt_generation.py"],
        check=True
    )

    print("\nPrompt generation completed!")

    print("\nSTEP 2: Generating images...\n")

    subprocess.run(
        [sys.executable, "image.py"],
        check=True
    )

    print("\nImage generation completed!")

    print("\nSTEP 3: Uploading...\n")    
    subprocess.run(
    [sys.executable, "upload_to_drive.py"],
    check=True
)

    print("\nPipeline completed successfully!")

except KeyboardInterrupt:
    print("\nExiting pipeline. User interruption")
    print("Done")

except subprocess.CalledProcessError as e:
    print(f"\nPipeline failed: {e}")

except Exception as e:
    print(f"\nUnexpected error: {e}")