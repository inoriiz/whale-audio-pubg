name: Build Windows EXE

# Runs automatically whenever you push code to the repo,
# and can also be triggered manually from the Actions tab.
on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest

    steps:
      - name: Check out repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pyaudiowpatch numpy customtkinter pyinstaller

      - name: Build EXE with PyInstaller
        run: |
          pyinstaller --onefile --windowed --name "WhaleAudioEnhancer" whale_audio_prototype.py

      # Uploads dist/WhaleAudioEnhancer.exe as a downloadable build artifact.
      # Find it under the finished workflow run's "Artifacts" section.
      - name: Upload EXE artifact
        uses: actions/upload-artifact@v4
        with:
          name: WhaleAudioEnhancer-windows-exe
          path: dist/WhaleAudioEnhancer.exe
