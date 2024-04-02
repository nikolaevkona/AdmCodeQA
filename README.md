### AdmCodeQA

Utils for QA about administrative fine

#### Example of use:
```
git clone https://github.com/nikolaevkona/AdmCodeQA.git
pip install -r AdmCodeQA/requirements.txt
wget -O coap.txt "https://www.dropbox.com/scl/fi/i56hj3za7ale9tiedaiza/coap.txt?rlkey=tjde2ubbfgx6rnsyn3mgksvqn&dl=1"
mkdir vector_store
python3 AdmCodeQA/run.py \
    --code-path coap.txt \
    --save-path vector_store \
    --api-key (your api key)
```