# getInterviews
Applying for jobs is boring, let getInterviews handle it.

## 🚀 Project Setup Checklist

- [x] **Clone and move to getInterviews repository**
    ```bash
    git clone https://github.com/pankajrawat9075/getInterviews.git
    cd getInterviews
    ```

- [x] **Clone browser_use fork and move into browser-use directory**
    ```bash
    git clone -b get_interviews_feature https://github.com/pankajrawat9075/browser-use.git
    cd browser-use
    ```

- [x] **Create and activate Virtual env**

    **CMD**
    ```bash
    pip install uv
    uv venv --python 3.11
    .venv\Scripts\activate.bat
    ```

    **Linux / Mac / WSL**
    ```bash
    uv venv --python 3.11
    source .venv/bin/activate
    ```
- [x] **Install dependencies**
    ```bash
    uv sync --all-extras
    playwright install chromium --with-deps --no-shell
    ```

- [x] **Install browser-use as editable package**
    ```bash
    cd ..
    uv pip install -e browser-use
    ```


