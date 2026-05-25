from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os


gauth = GoogleAuth()
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)


PROJECT_FOLDER_NAME = "AI_VIDEO_PIPELINE"

LOCAL_OUTPUT_FOLDER = "outputs"

JSON_FILE = r"D:/vs code/project/storyboard.json"


project_folder_metadata = {
    'title': PROJECT_FOLDER_NAME,
    'mimeType': 'application/vnd.google-apps.folder'
}

project_folder = drive.CreateFile(project_folder_metadata)
project_folder.Upload()

project_folder_id = project_folder['id']

print(f"Created project folder: {PROJECT_FOLDER_NAME}")


outputs_folder_metadata = {
    'title': 'outputs',
    'mimeType': 'application/vnd.google-apps.folder',
    'parents': [{'id': project_folder_id}]
}

outputs_folder = drive.CreateFile(outputs_folder_metadata)
outputs_folder.Upload()

outputs_folder_id = outputs_folder['id']

print("Created outputs folder")

json_drive_file = drive.CreateFile({
    'title': "storyboard.json",
    'parents': [{'id': project_folder_id}]
})

json_drive_file.SetContentFile(JSON_FILE)

json_drive_file.Upload()

print("Uploaded storyboard.json")


for filename in os.listdir(LOCAL_OUTPUT_FOLDER):

    filepath = os.path.join(LOCAL_OUTPUT_FOLDER, filename)

    if os.path.isfile(filepath):

        gfile = drive.CreateFile({
            'title': filename,
            'parents': [{'id': outputs_folder_id}]
        })

        gfile.SetContentFile(filepath)

        gfile.Upload()

        print(f"Uploaded: {filename}")

print("\nAll uploads completed!")