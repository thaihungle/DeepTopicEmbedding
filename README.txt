------Test big imdb data-------
////////Step0////////
install all python3 libs and build SDR if necessary
it is better to setup virtual environment
****command sysntax****
pip install -r requirements.txt
////////Step1////////
Download big imdb data
http://www.mediafire.com/file/pdzciu93zspaad1/data.json
///////Step2////////
Create hierarchical data
****command sysntax****
python big_imdb_prepare.py
<mode> must be PREPARE_HA
<max vocab word>
<max words in one sentence>
<max sentence in document>
<big imdb data file>
<output directory containing hierarchical data files>
****Example*****
python big_imdb_prepare.py PREPARE_HA 30000 200 50 ./data/data.json ./data/big_imdb_prep_ha/
///////Step3////////
Create term frequency data
****command sysntax****
python big_imdb_prepare.py
<mode> must be PREPARE_SDR
<output sdr train file name>
<output sdr test file name>
<input directory containing hierarchical data files>
****Example*****
 python big_imdb_prepare.py PREPARE_SDR ./data/big_imdb_count_data/big_imdb_raw_train.data ./data/big_imdb_count_data/big_imdb_raw_test.data ./data/big_imdb_prep_ha/
///////Step4////////
Run SDR OPE to get sdr word embedding
cd to two-phase-v2 directory
****command sysntax****
python sdr_ope.py
<infer algorithm>
<number of topics>
<input sdr train file name>
<input sdr test file name>
<model directory>
<OPE setting file>
<OPE algorithm>
<start sdr ope from what step>
****Example*****
python sdr_ope.py fstm 40 ../data/big_imdb_count_data/big_imdb_raw_train.data ../data/big_imdb_count_data/big_imdb_raw_test.data imdb ./OPE-master/settings.txt ML-FW 0
///////Step5////////
Run ha lstm to train data with sdr word embedding
****command sysntax****
python ha_lstm_batchfile.py
<input data dir>
<model dir>
<sdr embedding file>
<continue build from last save mode?>
<mode> must be build or eval
****Example*****
python ha_lstm_batchfile.py ./data/big_imdb_prep_ha/ ./models/ha_lstm_batchfile ./embfiles/fstm.30000.10.ha2.beta False build

