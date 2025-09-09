<div align="center">
    <h1>Graph Neural Network-Based DDoS Protection
    for Data Center Infrastructure</h1>
</div>

<h2>Overview</h2>
<p>This repository contains the implementation and experiments for the research paper <strong>"Graph Neural Network-Based DDoS Protection for Data Center Infrastructure"</strong>. Our work introduces a novel approach Graph Neural Network (GNN) based detection system which leverages Graph U-Nets to automatically classify and mitigate DDoS traffic.</p>


<h2>Datasets</h2>
<p>Our evaluation is conducted on three comprehensive DDoS datasets, providing diverse attack scenarios and network conditions:</p>

<table>
    <thead>
        <tr>
            <th>Dataset Name</th>
            <th>Source Paper</th>
            <th>Number of Features</th>
            <th>DDoS Flows</th>
            <th>Benign Flows</th>
            <th>Types of DDoS Attacks</th>
            <th>Download Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>CIC IDS 2017</strong></td>
            <td><a href="https://www.scitepress.org/papers/2018/66398/66398.pdf" target="_blank">Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization</a></td>
            <td>84</td>
            <td>128,025</td>
            <td>2,271,320</td>
            <td>LOIC</td>
            <td><a href="http://cicresearch.ca/CICDataset/CIC-IDS-2017/" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td><strong>CIC DDoS 2019</strong></td>
            <td><a href="https://ieeexplore.ieee.org/document/8888419" target="_blank">Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy</a></td>
            <td>86</td>
            <td>70,313,809</td>
            <td>113,828</td>
            <td>DNS; LDAP; MSSQL; NTP; SSDP; SNMP SYN Flood; UDP Flood; UDP-Lag; WebDDoS</td>
            <td><a href="http://cicresearch.ca//CICDataset/CICDDoS2019/" target="_blank">Link</a></td>
        </tr>
        <tr>
            <td><strong>BCCC CPacket Cloud DDoS 2024</strong></td>
            <td><a href="https://www.mdpi.com/2078-2489/15/4/195" target="_blank">Toward Generating a New Cloud-Based Distributed Denial of Service (DDoS) Dataset and Cloud Intrusion Traffic Characterization</a></td>
            <td>322</td>
            <td>228,469</td>
            <td>413,199</td>
            <td>TCP-SYN Variants; Control and KILLER TCP; Mixed Flag Attacks; Advanced SYN Attacks</td>
            <td><a href="https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/dataset-request/" target="_blank">Link</a></td>
        </tr>
    </tbody>
</table>

<h2>Models</h2>
<p>We compare our approach against state-of-the-art baseline models for Intrusion Detection:</p>

<table>
    <thead>
        <tr>
            <th>Model Name</th>
            <th>Graph Type</th>
            <th>Source Paper</th>
            <th>Type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>GNN Robust Network Intrusion Detection System </strong></td>
            <td>Host-Connection Graph</td>
            <td><a href="https://dl.acm.org/doi/abs/10.1145/3543146.3543171" target="_blank">Unveiling the potential of Graph Neural Networks for robust Intrusion Detection</a></td>
            <td><strong>Baseline</strong></td>
        </tr>
        <tr>
            <td><strong>E-GraphSage</strong></td>
            <td>Flow-Edge Graph</td>
            <td><a href="https://ieeexplore.ieee.org/abstract/document/9789878" target="_blank">E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT
</a></td>
            <td><strong>Baseline</strong></td>
        </tr>
        <tr>
            <td><strong>Heterogeneous Graph U-Nets</strong></td>
            <td>Temporal-Enhanced Host-Connection Graph</td>
            <td><a href="#" target="_blank">Graph Neural Network-Based DDoS Protection for Data Center Infrastructure (Ours)</a></td>
            <td><strong>Our Model</strong></td>
        </tr>
    </tbody>
</table>

<h2>Installation &amp; Setup</h2>

<h3>Step 1: Clone the Repository</h3>
<pre><code>git clone https://github.com/kartikeyas00/heterogeneous-graph-unets-ddos.git
cd heterogeneous-graph-unets-ddos</code></pre>

<h3>Step 2: Install Dependencies</h3>
<pre><code>pip install -r requirements.txt
</code></pre>

<h3>Step 3: Download Datasets</h3>
<pre>The datasets need to be downloaded from the link provided in the Dataset table. In some cases you need to contact the authors of the datataset papers to download the dataset.</pre>

<h2>Running the Pipeline</h2>

<h3>Complete Pipeline Execution</h3>
<p>The repository includes an automated pipeline script that handles data preprocessing, hyperparameter tuning, model training, and results collection. Run the pipeline for each dataset using the corresponding configuration file:</p>

<h4>CIC DDoS 2019 Dataset</h4>
<pre><code>cd cic-ddos-2019
python pipeline.py --config config.json</code></pre>

<h4>CIC IDS 2017 Dataset</h4>
<pre><code>cd cic-ids-2017
python pipeline.py --config config.json</code></pre>

<h4>BCCC CPacket Cloud DDoS 2024 Dataset</h4>
<pre><code>cd bccc-cpacket-cloud-ddos-2024
python pipeline.py --config config.json</code></pre>

<h3>Pipeline Steps</h3>
<p>Each pipeline execution includes the following automated steps:</p>
<ol>
    <li><strong>Data Preprocessing:</strong> Convert raw network flows into graph structures</li>
    <li><strong>Hyperparameter Tuning:</strong> Optimize model parameters using Optuna</li>
    <li><strong>Model Training:</strong> Train models with best hyperparameters using k-fold cross-validation</li>
    <li><strong>Results Collection:</strong> Aggregate and save performance metrics</li>
</ol>

<h3>Pipeline Options</h3>
<p>You can skip specific steps if needed:</p>
<pre><code>
# Skip preprocessing (if already done)
python pipeline.py --config config.json --skip-preprocessing
<br>
# Skip hyperparameter tuning (use existing parameters)
python pipeline.py --config config.json --skip-hyperparameter-tuning

<br>
# Skip training (only collect existing results)
python pipeline.py --config config.json --skip-training

<br>
# Custom log directory
python pipeline.py --config config.json --log-dir custom_logs</code></pre>

<h2>Customization &amp; Configuration</h2>

<h3>Pipeline Configuration</h3>
<p>Each dataset has its own configuration file that controls all aspects of the pipeline. Modify the <code>config.json</code> file in each dataset directory to customize the behavior:</p>

<h3>Preprocessing Settings</h3>
<p>Enable or disable preprocessing steps and adjust parameters in the config file:</p>
<pre><code>{
  "preprocessing": {
    "raw_data": {
      "enabled": true,
      "input_dir": "/path/to/raw/data/",
      "output_file": "/path/to/features.csv"
    },
    "host_connection_graph": {
      "enabled": true,
      "input_path": "/path/to/features.csv",
      "output_dir": "/path/to/graphs/host_connection_graph/",
      "class_ratio": 10.0,
      "k_folds": 10
    }
  }
}</code></pre>

<h3>Model Configuration</h3>
<p>Configure which models to run and their hyperparameter tuning settings:</p>
<pre><code>{
  "models": {
    "gnn_hetero_unet": {
      "enabled": true,
      "hyperparameter_tuning": {
        "optuna_dir": "/path/to/hyperparams/",
        "n_trials": 100,
        "study_name": "gnn_hetero_unet_hyperopt"
      },
      "training": {
        "epochs": 1000,
        "k_folds": 10,
        "patience": 50
      }
    }
  }
}</code></pre>

<h3>Training Parameters</h3>
<p>Adjust training settings for each model in the config file by modifying the training section for each model. Key parameters include:</p>
<ul>
    <li><strong>epochs:</strong> Maximum number of training epochs</li>
    <li><strong>k_folds:</strong> Number of cross-validation folds</li>
    <li><strong>patience:</strong> Early stopping patience</li>
    <li><strong>graph_dir:</strong> Directory containing graph data</li>
    <li><strong>log_dir:</strong> Directory for training logs</li>
</ul>

<h2>Citation</h2>
<p>If you use this code in your research, please cite our paper:</p>
<pre><code>@article{shar2025gnndata,
  title={Graph Neural Network-Based DDoS Protection for Data Center Infrastructure},
  author={Sharma, Kartikeya and Jacobik, Craig},
  booktitle={PNSQC Proceedings},
  pages={},
  year={2025}
}</code></pre>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <code>LICENSE</code> file for details.</p>
