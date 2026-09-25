# Natural Language Processing 13: Models and a Database in the Browser

This lab is a web page, not a notebook, so it does not run in Google Colab. It runs from a small local server on your own computer, with Python 3:

```
python3 fetch_web.py
NLPLAB_WEB=web python3 serve.py
```

`fetch_web.py` downloads the JavaScript libraries and the two ONNX models the page uses (about 315 MB) into `web/`; `serve.py` serves `app/` and them at http://localhost:8013/. Edit the files in `app/` and reload the page.
